# MAGIS: <ins>M</ins>emory Optimiz<ins>a</ins>tion via Coordinated <ins>G</ins>raph Transformat<ins>i</ins>on and <ins>S</ins>cheduling for DNN

[\[Paper\]](https://dl.acm.org/doi/10.1145/3620666.3651330) [\[Poster\]](./media/magis-asplos24-poster.pdf) [\[Slides\]](./media/magis-asplos24-talk.pdf)

## Install
```shell
python -m pip install -r requirements.txt
cd python && python3 setup.py [develop|install]
```
## Simple Example 
```python 
from magis.testing import nn
from magis.testing import setup_training_graph, run_optimization 

G, y = nn.bert("large", batch_size=64)
G = setup_training_graph(G, y, update_weight=True, inplace=False)

run_optimization(
	G,
	"name",
	mem_limit_ratio=0.8, # 80% memory ratio limit
	# lat_limit_ratio=1.1, # 10% latency overhead limit
	time_budget=3 * 60, # 3 minutes optimization time 
	save_graph=True, # save result graph + sched to "./data/name.pkl"
	dump_file=open("results.csv", "a"), # save profile result to "results.csv"
	dtype="float32", # data type 
)
```
The columns in `results.csv` represent: "name", "device memory limit", "latency limit", "memory limit", "latency limit ratio", "memory limit ratio", "weight memory", "opt-is-prof-result", "opt-latency", "opt-memory", "opt-simul-latency", "opt-simul-memory", "ori-is-prof-result", "ori-latency", "ori-memory", "ori-simul-latency", "ori-simul-memory"

Note that: 
- "memory" means peak-memory-usage (divided by data-type-bytes). 
- "opt" means "optimization" and "ori" means "origin". 
- "opt-is-prof-result" is True meaning that "opt-latency" and "opt-memory" are from real hardware profiling. 
- "opt-is-prof-result" is False meaning that "opt-latency" & "opt-memory" equals to "opt-simul-latency" & "opt-simul-memory", which are from simulation based on single-operator profiling results and memory analysis. 
- Generally, only "ori-is-prof-result" can be False since the original memory footprint may exceed the device memory limit. 
## Code Organization 
- `python/magis/`
	- `utils/`: Utilities for other components
		- `base_graph.py`: Basic graph data structure (using `rustworkx` library) for computation graph and dimension graph. 
		- `conv_utils.py`: Utilities for shape calculation of convolution 
		- `logging.py`: Utilities for logging 
		- `timing.py`: Utilities for recording python-code execution time 
		- `union_find_set.py`: Union-find-set data structure used for the construction of dimension graph 
	- `operators/`: Definitions of various operators 
	- `op_graph.py`: Computation graph (MAGIS Graph IR) 
	- `dim_graph.py`: Dimension graph 
	- `scheduler.py`: Schedulers to schedule the computation graph (with only re-reordering). 
	- `simulator.py`: Simulators to estimate the latency & memory based on the given computation graph and its schedule.  
	- `transform/`: Transformations 
		- `rewrite_rules/`: Pattern-match based rewriting rules
			- `base.py`: Basic definitions  
			- `taso_rules/`: Rules from TASO 
			- `sched_rules.py`: Rules derived from scheduling methods like Re-materialization and Swapping.
		- `misc.py`: Transformations other than rewriting rules
		- `mutator.py`: An abstraction of rewriting rules and other transformations. A mutator accepts a computation graph as input and generates a sequence of new graphs. Different mutators can be composited via combinators like "chain", "zip", "truncate" etc. 
	- `backend/`: Backends to compile/execute given graph + schedule 
		- `base.py`: Basic declarations. A backend provides interfaces to measure latency for a single operator and memory & latency for a whole graph.  A codegen backend additionally provides interfaces to generate code for each type of operators.
		- `torch_cuda.py`: A backend to generate python-code invoking PyTorch API. 
	- `optimizer.py`: Optimizer to optimize graph's memory & latency with the help of scheduler, simulator, and mutator.
	-  `testing/`: Utilities for testing 
		- `nn/`: Some neural networks defined using MAGIS Graph IR 
		- `bench.py`: Some utilities for running testing  
		- `config.py`: Some configurations 
 
We may provide more detailed explanations of our code in the future updates.

## Citation
If you find MAGIS useful or relevant to your project and research, please kindly cite our paper:
```bibtex
@inproceedings{10.1145/3620666.3651330,
	author = {Chen, Renze and Ding, Zijian and Zheng, Size and Zhang, Chengrui and Leng, Jingwen and Liu, Xuanzhe and Liang, Yun},
	title = {MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN},
	year = {2024},
	isbn = {9798400703867},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3620666.3651330},
	doi = {10.1145/3620666.3651330},
	pages = {607–621},
	numpages = {15},
	location = {<conf-loc>, <city>La Jolla</city>, <state>CA</state>, <country>USA</country>, </conf-loc>},
	series = {ASPLOS '24}
}
```
