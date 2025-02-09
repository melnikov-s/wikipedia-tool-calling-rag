import wiki from "wikipedia";
import { config } from "dotenv";
import { Document } from "@langchain/core/documents";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PromptTemplate } from "@langchain/core/prompts";
import { Annotation, MemorySaver, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { tool } from "@langchain/core/tools";
import chalk from "chalk";
import { z } from "zod";

const startingPrompt = `
You are an assistant for question-answering tasks specifically to wikipedia pages. 
The wikipedia page is provided as context, do not make up an answer, only use the context provided.
If you there is not enough information in the context, use the search tool named "wikipedia" to search wikipedia for more information.

Question: {question}
Answer:`;

const followUpPrompt = `
You are an assistant for question-answering tasks specifically to wikipedia pages. 

The user is asking a follow up question to the previous question.
The previous question was: {previousQuestion}
The previous context is provided below.
{context}

The previous answer is provided below.
{answer}

If you can answer the question based on the previous context and answer, do so.
Otherwise, use the search tool named "wikipedia" to search wikipedia for more information.

Question: {question}
Answer:`;

const prompt = `
Use the following pieces of retrieved context to answer the question.
 If you don't know the answer or it's not in the context, just say that you don't know. 
Question: {question} 
Context: {context} 
Answer:`;

const MAX_SEARCH_RESULTS = 3;

const wikipediaSchema = z.object({
  searchTerm: z.string().describe("The search term to search wikipedia for."),
});

async function main() {
  // Load environment variables
  config();

  // Verify API key is present
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is required in .env file");
  }

  const wikipediaTool = tool(getWikiPageContent, {
    name: "wikipedia",
    description: "Use this tool to search wikipedia",
    schema: wikipediaSchema,
  });

  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
    apiKey: process.env.OPENAI_API_KEY,
  }).bindTools([wikipediaTool]);

  const startingPromptTemplate = PromptTemplate.fromTemplate(startingPrompt);
  const followUpPromptTemplate = PromptTemplate.fromTemplate(followUpPrompt);
  const promptTemplate = PromptTemplate.fromTemplate(prompt);

  const InputStateAnnotation = Annotation.Root({
    question: Annotation<string>,
    context: Annotation<Document[]>,
    searchTerm: Annotation<string>,
    searchContent: Annotation<string>,
    answer: Annotation<string>,
    previousQuestion: Annotation<string>,
  });

  const StateAnnotation = Annotation.Root({
    question: Annotation<string>,
    context: Annotation<Document[]>,
    searchTerm: Annotation<string>,
    searchContent: Annotation<string>,
    answer: Annotation<string>,
    previousQuestion: Annotation<string>,
  });

  const search = async (state: typeof InputStateAnnotation.State) => {
    let messages;
    if (state.context && state.answer) {
      messages = await followUpPromptTemplate.invoke({
        question: state.question,
        context: state.context,
        answer: state.answer,
        previousQuestion: state.previousQuestion,
      });
    } else {
      messages = await startingPromptTemplate.invoke({
        question: state.question,
      });
    }

    const response = await llm.invoke(messages);

    const wikiToolCall = response.tool_calls?.find(
      (call) => call.name === "wikipedia"
    );

    if (!wikiToolCall) {
      return { searchTerm: "", answer: response.content };
    }

    const searchTerm = wikiToolCall.args.searchTerm;

    return {
      searchTerm,
      answer: response.content,
      previousQuestion: state.question,
    };
  };

  const fetchWiki = async (state: typeof StateAnnotation.State) => {
    if (!state.searchTerm) {
      console.log(chalk.dim("Answering based on previous context ...."));
      return state;
    }

    console.log(
      chalk.dim(`Fetching Wikipedia content for ${state.searchTerm}`)
    );

    const wikiContent = await getWikiPageContent({
      searchTerm: state.searchTerm,
    });

    return { searchContent: wikiContent };
  };

  const embeddingCache = new Map<string, MemoryVectorStore>();
  // Define application steps
  const retrieve = async (state: typeof StateAnnotation.State) => {
    let vectorStore = embeddingCache.get(state.searchContent);

    if (!vectorStore) {
      console.log(chalk.dim(`Creating embeddings...`));
      // Create vector store from the Wikipedia content
      vectorStore = await createEmbeddings(state.searchContent);
      embeddingCache.set(state.searchContent, vectorStore);
      console.log(
        chalk.dim(`Created embeddings stored in an in-memory vector store\n`)
      );
    } else {
      console.log(chalk.dim(`Using cached embeddings...`));
    }

    // Retrieve similar content
    const retrievedDocs = await vectorStore.similaritySearch(state.question);
    return { context: retrievedDocs };
  };

  const generate = async (state: typeof StateAnnotation.State) => {
    const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
    const messages = await promptTemplate.invoke({
      question: state.question,
      context: docsContent,
    });

    const response = await llm.invoke(messages);
    return { answer: response.content };
  };
  const memory = new MemorySaver();
  // Compile application and test
  const graph = new StateGraph(StateAnnotation)
    .addNode("search", search)
    .addNode("fetchWiki", fetchWiki)
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge("__start__", "search")
    .addEdge("search", "fetchWiki")
    .addEdge("fetchWiki", "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", "__end__")
    .compile({ checkpointer: memory });

  console.log(
    chalk.bold.yellow("\n📚 Ask questions about the Wikipedia page") +
      chalk.dim(" (type '/bye' to exit)\n")
  );

  process.stdin.setEncoding("utf-8");
  process.stdin.on("data", async (data) => {
    const question = data.toString().trim();

    if (question.toLowerCase() === "/bye") {
      console.log(chalk.bold.green("\nGoodbye! 👋\n"));
      process.exit(0);
    }

    if (question.length === 0) return;

    const inputs = { question };
    try {
      console.log(chalk.dim("\nThinking... 🤔"));
      const result = await graph.invoke(inputs, {
        configurable: { thread_id: 1 },
      });
      console.log(
        chalk.bold.blue("\n🤖 Answer:"),
        chalk.white(result.answer),
        "\n"
      );
    } catch (error) {
      console.error(
        chalk.bold.red("\n❌ Error processing question:"),
        chalk.red(error),
        "\n"
      );
    }
    console.log(
      chalk.yellow("Ask another question") +
        chalk.dim(" or type '/bye' to exit") +
        "\n"
    );
  });
}

async function createEmbeddings(text: string) {
  const textSplitter = new RecursiveCharacterTextSplitter();

  const allSplits = await textSplitter.createDocuments([text]);

  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-large",
    apiKey: process.env.OPENAI_API_KEY,
    openAIApiKey: process.env.OPENAI_API_KEY,
    verbose: true,
  });

  const vectorStore = new MemoryVectorStore(embeddings);

  await vectorStore.addDocuments(allSplits);

  return vectorStore;
}

async function getWikiPageContent({ searchTerm }: { searchTerm: string }) {
  console.log(chalk.dim(`Searching wikipedia for ${searchTerm}`));
  const searchResults = await wiki.search(searchTerm);

  const trimmedResults = searchResults.results
    .slice(0, MAX_SEARCH_RESULTS)
    .map((result) => result.title);

  const contents = await Promise.all(
    trimmedResults.map(async (title) => {
      const page = await wiki.page(title);
      return page.content();
    })
  );

  console.log(
    chalk.dim(
      `Wikipedia search complete, found ${trimmedResults
        .map((title) => title)
        .join(", ")} results`
    )
  );

  return contents.join("\n");
}

main();
