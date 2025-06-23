
from dataclasses import dataclass, asdict
import requests

@dataclass
class CSSLanguageParameters:
	classes: list[str]
	ids: list[str]

@dataclass
class TacoScheduleParameters:
	index_variables: list[str]
	workspace_index_variables: list[str]
	fused_index_variables: list[str]
	split_factor_variables: list[str]
	divide_factor_variables: list[str]
	unroll_factor_variables: list[str]

@dataclass
class TacoExpressionParameters:
	symbols: list[str]
	indices: list[str]

@dataclass
class GenerateParams:
	count: int
	return_edge_lists: bool
	return_features: bool
	return_grammar: bool
	wl_degree: int

	css: CSSLanguageParameters
	# taco_schedule: TacoScheduleParameters
	# taco_expression: TacoExpressionParameters

@dataclass
class GenerateResults:
	edge_lists: list
	features: list
	grammar: str
	programs: list

def generate(url: str, language: str, expander: str, params: GenerateParams):
	url = f"{url}/v1/generate/{language}/{expander}"

	res = requests.post(url, json=asdict(params), headers={"Content-Type": "application/json"})

	if res.status_code != 200:
		print(f"unsuccessful response: {res}")
		return

	return GenerateResults(**res.json())
