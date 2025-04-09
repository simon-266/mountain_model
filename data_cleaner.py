import pandas as pd
import ollama
from io import StringIO

def clean_data(df: pd.DataFrame, target_columns: list, sample_data: pd.DataFrame = None, additional_prompt: str = "", model: str = "deepseek-r1", chunk_size: int = 200):
    """
    Cleans data in a Pandas DataFrame using the DeepSeek language model via Ollama.

    Please consider this script expects a running ollama server and the selected model to be installed
    for further information consider the official website of ollama https://ollama.com/

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        target_columns (list): A list of column names to extract and clean.
        sample_data (pd.DataFrame, optional): A small sample of the desired output format. Defaults to None.
        additional_prompt (str, optional): Additional instructions for DeepSeek. Defaults to "".
        model (str, optional): The DeepSeek model to use (must be available in Ollama). Defaults to "deepseek-r1".
        chunk_size (int, optional): The number of rows to process in each chunk. Defaults to 200.

    Returns:
        pd.DataFrame: The cleaned DataFrame, or an empty DataFrame if errors occur.
    """
    cleaned_chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df[i:i + chunk_size]
        data_text = chunk.to_csv(index=False)

        prompt = f"""You are a data processing expert whose sole task is to convert completely inconsistent data into a clean CSV format. You will receive a set of unstructured data containing information about various entities. Your task is to extract the relevant information for the following columns and output it in a standardized CSV format.

        **Important Rules:**

        1.  **Limit to the specified columns:** Extract and use only the columns defined here: `{','.join(target_columns)}`. Ignore all other information in the input data.
        2.  **Clean CSV format:** The output MUST be a valid CSV format. This means:
            * The first line contains the exact column names: `{','.join(target_columns)}`, separated by commas.
            * Each subsequent line represents a data record, with the values in the same order as the column names and separated by commas.
            * There must be no extra spaces before or after the commas or the values.
        3.  **No additional output:** Output ONLY the CSV data and the header row. Any other text, explanations, greetings, or additional information are strictly prohibited.
        4.  **Handle inconsistencies:** Assume that the input data can be highly inconsistent. Information for a specific column may be missing, formatted differently, or appear in unexpected places. Do your best to identify and map the relevant information. If information for a required column is completely missing in a record, leave the corresponding field in the CSV output empty.
        5.  **Direct output:** Start the output directly with the header row, followed by the data rows.

        {'**Example of desired output format (if provided):**' if sample_data is not None else ''}
        {sample_data.to_csv(index=False) if sample_data is not None else ''}

        **Inconsistent Input Data:**

        {data_text}

        {additional_prompt}
        """
    
        try:
            response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            cleaned_chunk = pd.read_csv(pd.compat.StringIO(response["message"]["content"]))

            if list(cleaned_chunk.columns) != target_columns:
                print(f"Warning: Output columns do not match target columns. Output columns: {cleaned_chunk.columns}, Target columns: {target_columns}")
                cleaned_chunk = cleaned_chunk.reindex(columns=target_columns)

            cleaned_chunks.append(cleaned_chunk)
        except Exception as e:
            print(f"Error processing a chunk: {e}")
            print(f"Prompt was: {prompt}")
            print(f"Response was: {response}")
            pass

    if cleaned_chunks:
        cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
        return cleaned_df
    else:
        return pd.DataFrame()

def main():
    df = pd.read_csv("Mountain.csv")
    df_sample = pd.read_csv("unfilled.csv")
    df_sample = df_sample.dropna()
    df = clean_data(df, target_columns=["name", "height", "mountainRange"], sample_data=df_sample)
    df.head()


if __name__ == "__main__":
    main()