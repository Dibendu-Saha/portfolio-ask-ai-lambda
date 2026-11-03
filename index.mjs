import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";

export const handler = async (event) => {
  const headers = {
    "Access-Control-Allow-Origin": "*",
    "Content-Type": "application/json",
  };

  try {
    const { OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY } = process.env;
    if (!OPENAI_API_KEY || !SUPABASE_URL || !SUPABASE_KEY) {
      return {
        statusCode: 500,
        headers,
        body: JSON.stringify({ error: "Missing environment variables." }),
      };
    }

    const body = event.body ? JSON.parse(event.body) : {};
    const userQuery = body.question?.trim();
    if (!userQuery) {
      return {
        statusCode: 400,
        headers,
        body: JSON.stringify({ error: "Payload property 'question' missing." }),
      };
    }

    const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
    const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

    // 1. Refine user query
    const queryResponse = await openai.responses.create({
    model: "gpt-4o-mini",
    instructions: "ONLY refine the provided user query so that it is clear and understable.",
    input: userQuery
  });

  const refinedUserQuery = queryResponse.output_text;

    // 2. Vector embedding
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: refinedUserQuery,
    });
    const queryEmbedding = embeddingResponse.data[0].embedding;

    // 3. Supabase similarity search
    const searchResponse = await supabase.rpc("match_documents", {
      query_embedding: queryEmbedding,
      match_threshold: 0.2,
      match_count: 3,
    });

    if (searchResponse.error) {
      console.error("Supabase RPC error:", searchResponse.error);
      throw new Error("Supabase query failed");
    }

    const searchResult = (searchResponse.data || []).map((match) => match.content);
    const context = `Context:\n${searchResult}`;

    // 4. OpenAI Response API
    const response = await openai.responses.create({
      model: "gpt-4o-mini",
      instructions: 
        `You are Dibendu Saha's portfolio assistant. 
        ONLY answer based on the content provided against the 'Context:' block.
        Act like you're him, and provide first-person responses in a natural tone. 
        If not in the context, politely indicate that the information is not available and suggest contacting Dibendu directly.`,
      input: `${context}\n\n${refinedUserQuery}`,
    });

    const answer = response.output_text || "No response received.";

    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ answer }),
    };
  } catch (error) {
    console.error("Handler error:", error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: "Internal server error." }),
    };
  }
};
