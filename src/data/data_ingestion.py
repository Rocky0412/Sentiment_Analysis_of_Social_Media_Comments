import pandas as pd # type: ignore
import requests # type: ignore
import certifi # type: ignore
import numpy as np # type: ignore
import io
import logging 
import os

# ----------------------- LOGGER SETUP -----------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    fh = logging.FileHandler("logs.log")
    ch = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

# ----------------------- DATA INGESTION -----------------------
def data_ingestion(url: str = None) -> pd.DataFrame:
    if url is None:
        raise ValueError("URL cannot be None.")

    df = None
    try:
        r = requests.get(url, verify=certifi.where(), timeout=10)
        r.raise_for_status()
        logger.info("Request completed successfully.")

        try:
            df = pd.read_csv(io.StringIO(r.text))
            os.makedirs("data/raw", exist_ok=True)

            # Cleaning steps
            df.dropna(subset=['category'], inplace=True)
            df.drop_duplicates(inplace=True)

            df['clean_comment'] = df['clean_comment'].astype(str)
            df['clean_comment'] = df['clean_comment'].str.strip()
            df['clean_comment'] = df['clean_comment'].fillna(" ")

            df.to_csv("data/raw/raw_data.csv", index=False)
            logger.info(f"Data saved successfully. Shape: {df.shape}")

        except Exception as e:
            logger.error(f"Error while reading or processing CSV: {e}", exc_info=True)

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

    logger.info("Data ingestion completed.")
    return df

# ----------------------- MAIN -----------------------
if __name__ == "__main__":
    logger.info("Data Ingestion Started")
    url = "https://raw.githubusercontent.com/Rocky0412/Datasets/master/Reddit_Data.csv"
    df = data_ingestion(url)
