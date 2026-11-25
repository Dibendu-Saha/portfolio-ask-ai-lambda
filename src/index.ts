import "dotenv/config";
import fs from "node:fs";
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";
import { Client, Tool } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";

export const handler = async (event: any) => {
  const headers = {
    "Access-Control-Allow-Origin": "*",
    "Content-Type": "application/json",
  };

  try {
    // init MCP client & get required tools from MCP server
    const { tools, mcpClient } = await initMcpClient();

    const { OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY } = process.env;
    if (!OPENAI_API_KEY || !SUPABASE_URL || !SUPABASE_KEY) {
      return {
        statusCode: 500,
        headers,
        body: JSON.stringify({ error: "Missing environment variables." }),
      };
    }

    const body = event.body ? JSON.parse(event.body) : {};
    const userQuery: string | undefined = body.question?.trim();
    if (!userQuery) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: "Payload property 'question' missing." }),
      };
    }

    const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
    const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

    /****************************************************
     *************** 1. refine user query ***************
     ****************************************************/
    const queryResponse = await openai.responses.create({
      model: "gpt-4o-mini",
      instructions: "ONLY refine the provided user query so that it is clear and understable.",
      input: userQuery,
    });

    const refinedUserQuery = queryResponse.output_text;

    /****************************************************
     *************** 2. vector embeddings ***************
     ****************************************************/
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: refinedUserQuery,
    });
    const queryEmbedding = embeddingResponse.data[0].embedding;

    /****************************************************
     ********** 3. supabase similarity search ***********
     ****************************************************/
    const searchResponse = await supabase.rpc("match_documents", {
      query_embedding: queryEmbedding,
      match_threshold: 0.2,
      match_count: 3,
    });

    if (searchResponse.error) {
      console.error("Supabase RPC error:", searchResponse.error);
      throw new Error("Supabase query failed");
    }

    const searchResult = (searchResponse.data || []).map((match: any) => match.content);
    const context = `Context:\n${searchResult}`;

    /****************************************************
     ************** 4. OpenAI Response API **************
     ****************************************************/
    const response = await openai.responses.create({
      model: "gpt-4o-mini",
      instructions:
        `You are Dibendu Saha's portfolio assistant. 
        ONLY answer based on the content provided against the 'Context:' block.
        Act like you're him, and provide first-person responses in a natural tone. 
        If not in the context, politely indicate that the information is not available and suggest contacting Dibendu directly.
        
        You also have access to the 'readGoogleDoc' tool to read his latest CV.
        The CV Google Document ID is: ${process.env.GOOGLE_DOC_ID}.
        If the user asks about Dibendu's CV, resume, or career history, call 'readGoogleDoc' using this document ID.`,
      input: `${context}\n\n${refinedUserQuery}`,
      tools
    });

    let answer;
    console.log("\n--- OpenAI Raw Output ---");
    console.log(JSON.stringify(response.output || response, null, 2));


    let nextResponse;
    if (response.output[0].type === "function_call") {
      // 1. execute the tool call
      const result = await mcpClient.callTool({
        name: response.output[0].name,
        arguments: JSON.parse(response.output[0].arguments)
      })

      if (result.isError) {
        console.error("Error retrieving tool response.");
      } else {
        // 2. get response
        console.log(result.content)

        // 3. hand-off to OpenAI
        nextResponse = await openai.responses.create({
          model: "gpt-4o-mini",
          previous_response_id: response.id,
          input: [{
            type: "function_call_output",
            call_id: response.output[0].call_id,
            output: JSON.stringify(result.content),
          }],
          store: true
        });

        answer = nextResponse!.output_text || "No response received.";
      }

    } else {
      answer = response!.output_text || "No response received.";
    }

    await mcpClient.close();

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ answer }),
    };
  } catch (error: any) {
    console.error("Handler error:", error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: "Internal server error." }),
    };
  }
};

const initMcpClient = async () => {
  class MCPClient {
    private mcpClient: Client;
    private transport: StdioClientTransport | null = null;
    private tools: Tool[] = [];

    constructor() {
      this.mcpClient = new Client(
        { name: "portfolio-ask-ai-client", version: "1.0.0" },
        { versionNegotiation: { mode: "auto" } },
      );
    }

    get getMcpClient() {
      return this.mcpClient;
    }

    async connectToMcpServer() {
      const CREDENTIAL_PATH = "/tmp/service-account.json";

      // write GCP credential json to AWS Lambda path --> read on next step from this path
      // if (!fs.existsSync(CREDENTIAL_PATH) && process.env.GCP_SERVICE_ACCOUNT_JSON) {
      //   fs.writeFileSync(CREDENTIAL_PATH, process.env.GCP_SERVICE_ACCOUNT_JSON, "utf8");
      // }

      this.transport = new StdioClientTransport({
        command: "npx",
        args: ["-y", "@piotr-agier/google-drive-mcp"],
        env: {
          ...process.env,
          GOOGLE_APPLICATION_CREDENTIALS:
            process.env.GOOGLE_APPLICATION_CREDENTIALS || CREDENTIAL_PATH,
        }
      });

      console.log("Connecting to @piotr-agier/google-drive-mcp...\n");
      await this.mcpClient.connect(this.transport);

      const toolsResult = await this.mcpClient.listTools();

      this.tools = toolsResult.tools.filter((tool) =>
        ["readGoogleDoc", "getGoogleDocContent"].includes(tool.name)
      );

      return this.tools.map(tool => ({
        type: "function" as const,
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema,
        strict: false,
      }));
    }
  }

  const client = new MCPClient();
  const mcpClient = client.getMcpClient;
  const tools = await client.connectToMcpServer();
  return { tools, mcpClient };
}

handler({
  body: JSON.stringify({
    question: "Can you summarize Dibendu's latest experience from his CV Google Doc?",
  }),
})
  .then((response: any) => {
    console.log("\n================ LAMBDA RESPONSE ================");
    console.log("Status Code:", response.statusCode);
    console.log("Body:", JSON.parse(response.body));
    console.log("=================================================");
  })
  .catch(console.error);