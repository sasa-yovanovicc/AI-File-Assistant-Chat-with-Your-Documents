param(
  [Parameter(Mandatory=$true)][string]$InputPath,
  [int]$Batch = 64
)
python -m src.ingest --input "$InputPath" --batch $Batch
