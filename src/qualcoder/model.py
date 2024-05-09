import pandas as pd
from typing import Literal, Union
from dataclasses import dataclass
import ollama


MODEL = 'coding-model'

Example = tuple[str, str]
Parameters = dict[str, Union[str, int, float]]


@dataclass
class ModelSettings:
    examples: list[Example]
    parameters: Parameters
    from_model: str
    name: str
    system: str

    def wirte_modelfile(self, out_file_location: str) -> str:
        """Creates modelfile and returns it as a string

        See documentation at https://github.com/ollama/ollama/blob/main/docs/modelfile.md
        """
        modelfile_contents = f"FROM {self.from_model}\n"
        if len(self.parameters) != 0:
            for p, val in self.parameters.items():
                modelfile_contents += f"PARAMETER {p} {val}\n"

        if len(self.examples) != 0:
            for prompt, answer in self.examples:
                modelfile_contents += f'MESSAGE user """{prompt}"""\n'
                modelfile_contents += f'MESSAGE assistant """{answer}"""\n'

        modelfile_contents += f'\nSYSTEM """\n{self.system}\n"""'

        with open(f'{out_file_location}{self.name}-modelfile', 'w+') as f:
            f.write(modelfile_contents)

        return modelfile_contents


def read_model_folder(folder: str = '../models/example') -> ModelSettings:

    pass


def read_parameters(folder: str) -> Parameters:
    params = pd.read_csv(f'{folder}/parameters.csv')\
        .set_index('parameter')\
        .to_dict('index')

    return {p: val['value'] for p, val in params}


def read_examples(folder: str) -> list[Example]:
    examples: list[Example] = []
    df = pd.read_csv(f'{folder}/examples.csv')

    for r in df.iterrows():
        examples.append(tuple(r['user'], r['assistant']))

    return [tuple(r['user'], r['assistant']) for r in df.iterrows()]


def create_model(modelfile: str, modelname=MODEL):
    ollama.create(model=modelname, modelfile=modelfile)
