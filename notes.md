# Ramdom notes

## Random ideas

- [x] add .gitignore file in git repositories. gitignore was not a hidden file. 
- add documentation
- use BetterBibtex plugin in Zotero to create .bib file to have all metadata about PDFs. Maybe explore if possible to connect directly to Zotero DB (in read mode).
- review code structure, tools/intention
- Add export/import functionality (if possible) for the embedding/parsed DB
- Create the embedding/parsing DB outside of the repository. Maybe somewhere like ~/.local/share/refchat/
- Add API keys and else in a hidden file in Home/


## PDF parsing
can we limit the use of grobid and use somthign like https://www.npmjs.com/package/pdf-parse to parse text and images from pdf as a first guess?
The question is about figuringout if there are steps no requiring grobid heavy processing and that a first parsing using somehting more deterministic may work.

Why even use grobid to retreive information that Zotero can provide as a bibtext file? And extracting text from PDF can be done with simpler tools:
- https://github.com/pdfminer/pdfminer.six
- https://medium.com/@bpmcgough/comparing-pdf-parsers-1b9f5ae24afe

Maybe having two tier pdf screening. One quick and dirty with simple tools. And then identify problematic parsing and use more complex parsing method using deep learning. Would it also be possible to outsource the PDF parsing to Mistral servers or equivalent?

## Documentation
- add ./doc/ to build a Sphynx documentation
- write documentation
- write roadmap

## modif
- changed the localhost port to 8001 that is free on linux. Maybe Windows app uses 5001 by default, so make a if statement based on OS type. The Python package `os` can check what OS (windos, linux, mac) is the computer. 
- added elements to linux install
- stop ollama and docker services when exiting. Start them on start up. 



