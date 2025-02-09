# wikipedia-tool-calling-rag

A simple example using LangChain + LangGraph to make a tool calling RAG application that answers questions using wikipedia as a source.

```
bun run index.ts

📚 Ask questions about the Wikipedia page (type '/bye' to exit)

Who is Batman?

Thinking... 🤔
Fetching Wikipedia content for Batman
Searching wikipedia for Batman
Wikipedia search complete, found Batman, Batman in film, The Batman (film) results
Creating embeddings...
Created embeddings stored in an in-memory vector store


🤖 Answer: Batman is a superhero who appears in American comic books published by DC Comics. His secret identity is Bruce Wayne, a wealthy American playboy, philanthropist, and industrialist residing in Gotham City. Batman was created by artist Bob Kane and writer Bill Finger, debuting in Detective Comics #27 on March 30, 1939. His origin story involves witnessing the murder of his parents, which drives him to seek justice against criminals. He trains himself physically and intellectually, adopts a bat-inspired persona, and fights crime in Gotham City. Batman is known for his iconic status in popular culture and has been portrayed by various actors in films and television series. 

Ask another question or type '/bye' to exit

What movies featured him?

Thinking... 🤔
Fetching Wikipedia content for Batman movies
Searching wikipedia for Batman movies
Wikipedia search complete, found Batman in film, Batman Returns, Batman Begins results
Creating embeddings...
Created embeddings stored in an in-memory vector store


🤖 Answer: The movies that featured Batman include:

1. Batman (1943) - Serial film
2. Batman and Robin (1949) - Serial film
3. Batman (1966) - Feature film adaptation of the 1960s television series
4. Batman (1989) - Directed by Tim Burton, starring Michael Keaton
5. Batman Returns (1992) - Directed by Tim Burton, starring Michael Keaton
6. Batman Forever (1995) - Directed by Joel Schumacher, starring Val Kilmer
7. Batman & Robin (1997) - Directed by Joel Schumacher, starring George Clooney
8. Batman v Superman: Dawn of Justice (2016) - Starring Ben Affleck
9. Justice League (2017) - Starring Ben Affleck
10. The Batman (2022) - Directed by Matt Reeves, starring Robert Pattinson
11. The Flash (2023) - Featuring Michael Keaton and Ben Affleck reprising their roles as Batman

Additionally, Batman made cameo appearances in various other DCEU films. 

Ask another question or type '/bye' to exit

Who played Batman in The Flash from 2023? 

Thinking... 🤔
Answering based on previous context ....
Using cached embeddings...

🤖 Answer: In The Flash (2023), Batman was played by Michael Keaton and Ben Affleck. 

Ask another question or type '/bye' to exit

Who won the worldcup in 2014?

Thinking... 🤔
Fetching Wikipedia content for 2014 FIFA World Cup
Searching wikipedia for 2014 FIFA World Cup
Wikipedia search complete, found 2014 FIFA World Cup, 2014 FIFA World Cup final, 2014 FIFA World Cup squads results
Creating embeddings...
Created embeddings stored in an in-memory vector store


🤖 Answer: Germany won the World Cup in 2014. 

```