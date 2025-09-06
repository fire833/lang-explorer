
from dataclasses import dataclass, asdict
import requests

@dataclass
class CSSLanguageParameters:
	version: str

	classes: list[str]
	ids: list[str]
	colors: list[str]

@dataclass
class TacoScheduleParameters:
	version: str

	index_variables: list[str]
	workspace_index_variables: list[str]
	fused_index_variables: list[str]
	split_factor_variables: list[str]
	divide_factor_variables: list[str]
	unroll_factor_variables: list[str]

@dataclass
class TacoExpressionParameters:
	version: str

	symbols: list[str]
	indices: list[str]

@dataclass
class KarelLanguageParameters:
	pass

@dataclass
class GeneralTrainingParameters:
	batch_size: int
	n_epochs: int
	learning_rate: float
	learning_rate_drop: float
	seed: int
	display_frequency: int
	create_new_model: bool
	save_model: bool

@dataclass
class GenerateParams:
	count: int
	return_edge_lists: bool
	return_features: bool
	return_graphviz: bool
	return_grammar: bool
	return_partial_graphs: bool
	return_embeddings: list[str]
	wl_degree: int
	num_neg_samples: int
	d_model: int
	window_left: int
	window_right: int
	gradient_clip_norm: float
	agg: str
	gen_params: GeneralTrainingParameters

	css: CSSLanguageParameters
	taco_schedule: TacoScheduleParameters
	taco_expression: TacoExpressionParameters
	karel: KarelLanguageParameters

@dataclass
class ProgramResult:
	program: str
	graphviz: str
	embeddings: dict
	features: list[int]
	edge_lists: list[str]
	is_partial: bool

@dataclass
class GenerateResults:
	programs: list[ProgramResult]
	grammar: str

def generate(url: str, language: str, expander: str, params: GenerateParams):
	url = f"{url}/v2/generate/{language}/{expander}"

	res = requests.post(url, json=asdict(params), headers={"Content-Type": "application/json"})

	if res.status_code != 200:
		print(f"unsuccessful response: {res}")
		return

	return GenerateResults(**res.json())
