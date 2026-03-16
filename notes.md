# Ramdom notes

## PDF parsing
can we limit the use of grobid and use somthign like https://www.npmjs.com/package/pdf-parse to parse text and images from pdf as a first guess?
The question is about figuringout if there are steps no requiring grobid heavy processing and that a first parsing using somehting more deterministic may work.

Why even use grobid to retreive information that Zotero can provide as a bibtext file? And extracting text from PDF can be done with simpler tools:
- https://github.com/pdfminer/pdfminer.six
- https://medium.com/@bpmcgough/comparing-pdf-parsers-1b9f5ae24afe

## Documentation
- add ./doc/ to build a Sphynx documentation
- write documentation
- write roadmap

## modif
- changed the localhost port to 8001 that is free on linux. Maybe Windows app uses 5001 by default, so make a if statement based on OS type. 
- added elements to linux install
- stop ollama and docker services when exiting. Start them on start up 


