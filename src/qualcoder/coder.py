import pandas as pd
import ollama

def code(df: pd.DataFrame, modelname: str, strategy) -> list[str]:
    """Codes given dataset. The dataset should have time and text columns.

    :param pd.DataFrame df: dataframe with time and text columns
    :param str modelname: name of the model to be used
    :param callable[pd.Dataframe, callable[str,str]] strategy: function returned from `GenerateStrategy` class' methods
    :return list[str]: list of responses from the model
    """
    results: list[str] = []
    run_strategy = strategy(df, modelname)
    for row in df.iterrows():
        results.append(run_strategy(row))
    return results


def join_time_text(row: pd.Series):
    return f'{row['time']} {row['text']}'


class GenerateStrategy:
    @staticmethod
    def line_by_line(df: pd.DataFrame, modelname: str):
        history = []
        def fun(row: pd.Series):
            history.append({'role':'user','content':join_time_text(row)})
            response = ollama.chat(model=modelname,messages=history)
            history.append(response['message'])
            return response['message']['content']
        return fun