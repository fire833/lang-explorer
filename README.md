
![Lang-explorer framework](images/langexplorerframework.svg)

# Lang Explorer

Lang-explorer is a framework for writing programs using [formal grammars](https://en.wikipedia.org/wiki/Formal_grammar). It was originally authored as the software framework for the research conducted in my Master's thesis ([defense slides](https://static.tauser.us/thesis/thesis_presentation.pdf), and [document](https://static.tauser.us/thesis/thesis.pdf)). As a result of this, it is likely a little rough around the edges. More work is being done to improve it's capabilities and to make it easier for others to use. Any suggestions or PRs are welcome.

### Learned Expander

![Learned Expander](images/learnedexpander.svg)

### Building/Compiling C++

```bash
# WD: Root of this repository
cmake .
make # optionally provide -j<number> to compile on multiple cores, probably not needed though
```
