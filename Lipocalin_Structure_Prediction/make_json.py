import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import typer

app = typer.Typer(
    help="CLI to convert FASTA files to SeedFold JSON job templates.",
    add_completion=False,
)

DEFAULT_MODEL = "SeedFold_v1.0.0"
DEFAULT_ENTITY_TYPE = "Protein"


def parse_fasta(fasta_path: Path) -> List[Tuple[str, str]]:
    """Parses a FASTA file and extracts sequence names and sequences.

    Args:
        fasta_path (Path): The path to the input FASTA file.

    Returns:
        List[Tuple[str, str]]: A list of tuples where the first element is the
            sequence name (header without the '>' character) and the second
            element is the continuous sequence string.

    Raises:
        ValueError: If the file is empty or improperly formatted.
    """
    sequences: List[Tuple[str, str]] = []
    current_name = ""
    current_seq: List[str] = []

    with fasta_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith(">"):
                if current_name:
                    # Save the previous sequence before starting a new one
                    sequences.append((current_name, "".join(current_seq)))
                # Extract the name, splitting at the first space to get the ID
                current_name = line[1:].strip().split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        # Append the final sequence in the file
        if current_name:
            sequences.append((current_name, "".join(current_seq)))

    if not sequences:
        raise ValueError(f"No valid FASTA sequences found in {fasta_path}")

    return sequences


def build_job_config(
    job_name: str,
    sequence: str,
    model_name: str,
    entity_type: str,
    copies: int,
) -> Dict[str, Any]:
    """Constructs a single job configuration dictionary.

    Args:
        job_name (str): The name of the job, typically the protein name.
        sequence (str): The biological sequence string.
        model_name (str): The name of the prediction model to use.
        entity_type (str): The type of the entity (e.g., 'Protein', 'DNA').
        copies (int): The number of copies of this entity in the job.

    Returns:
        Dict[str, Any]: A dictionary formatted for the job JSON array.
    """
    return {
        "job_name": job_name,
        "model": model_name,
        "entities": [
            {
                "entity": entity_type,
                "copies": copies,
                "sequence": sequence,
            }
        ],
    }


def generate_jobs_json(
    fasta_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL,
    entity_type: str = DEFAULT_ENTITY_TYPE,
    copies: int = 1,
) -> None:
    """Reads FASTA data, builds job configurations, and writes to a JSON file.

    Args:
        fasta_path (Path): Path to the input FASTA file.
        output_path (Path): Path to the output JSON file.
        model_name (str): The name of the model to use for prediction.
        entity_type (str): The entity type string.
        copies (int): Number of copies of the sequence per job.
    """
    fasta_data = parse_fasta(fasta_path)

    jobs = []
    for name, seq in fasta_data:
        job = build_job_config(
            job_name=name,
            sequence=seq,
            model_name=model_name,
            entity_type=entity_type,
            copies=copies,
        )
        jobs.append(job)

    with output_path.open("w", encoding="utf-8") as out_file:
        json.dump(jobs, out_file, indent=2)


@app.command()
def main(
    fasta_file: Path = typer.Argument(
        ...,
        help="Path to the input .fasta file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_file: Path = typer.Argument(
        ...,
        help="Path where the output .json file will be saved.",
        writable=True,
    ),
    model: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
        help="The model name to use for predictions.",
    ),
    entity_type: str = typer.Option(
        DEFAULT_ENTITY_TYPE,
        "--entity-type",
        "-e",
        help="The entity type for the sequences.",
    ),
    copies: int = typer.Option(
        1,
        "--copies",
        "-c",
        help="Number of copies for the entity in each job.",
        min=1,
    ),
) -> None:
    """
    Convert a FASTA file of proteins into a JSON job configuration file.
    """
    try:
        generate_jobs_json(
            fasta_path=fasta_file,
            output_path=output_file,
            model_name=model,
            entity_type=entity_type,
            copies=copies,
        )
        typer.echo(f"Successfully generated '{output_file}' with {copies} copy/copies per job.")
    except Exception as e:
        typer.echo(f"Error processing file: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
