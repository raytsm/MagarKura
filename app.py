from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fairseq.models.transformer import TransformerModel
import io
import csv

# ----------------------
# Load your Fairseq model
# ----------------------
MODEL_DIR = "/Users/roshanthapa/WD/testFairSeq/magarkura_v3/"

model = TransformerModel.from_pretrained(
    MODEL_DIR,
    checkpoint_file="checkpoint_best.pt",
    source_lang="np",
    target_lang="mgr",
    bpe="sentencepiece",
    sentencepiece_model="spm-twelve-hundred.model"
)

print(model.translate("<2mgr>मेरो नाम रोशन थापा हो।"))  # test translation

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI()

# Single sentence input
class InputText(BaseModel):
    text: str
    direction: str

@app.post("/translate")
async def translate(input: InputText):
    if input.direction == "np2mgr":
        input_text = "<2mgr>" + input.text
    elif input.direction == "mgr2np":
        input_text = "<2np>" + input.text
    else:
        return {"error": "Invalid direction"}
    output = model.translate(input_text)
    return {"translation": output}

# Bulk translation from TXT file
@app.post("/bulk_translate")
async def bulk_translate(
    file: UploadFile = File(...),
    direction: str = Form(...)
):
    """
    Accepts a .txt file with one sentence per line and returns a CSV
    with columns: input, translation
    """
    # Read all lines from the uploaded file
    contents = await file.read()
    lines = contents.decode("utf-8").splitlines()

    # Prepare CSV in memory
    output_csv = io.StringIO()
    writer = csv.writer(output_csv, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["input", "translation"])  # CSV header

    # Translate each line and write directly to CSV
    for line in lines:
        if direction == "np2mgr":
            input_text = "<2mgr>" + line
        elif direction == "mgr2np":
            input_text = "<2np>" + line
        else:
            return {"error": "Invalid direction"}
        
        translated = model.translate(input_text)
        writer.writerow([line, translated])

    output_csv.seek(0)

    # Return CSV as a downloadable file
    return StreamingResponse(
        io.BytesIO(output_csv.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=translations.csv"}
    )