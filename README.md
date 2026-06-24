# Assignment 1: Word Embedding API

This project extends the Module 2 FastAPI text generation API by adding a spaCy word embedding endpoint.

## Features

* `GET /` returns a welcome message.
* `POST /generate` generates text using a simple bigram model.
* `GET /embedding?word=<word>` returns the spaCy word embedding for a query word.

## Example

Request:

```text
GET /embedding?word=happy
```

Example response:

```json
{
  "word": "happy",
  "vector_dimension": 300,
  "embedding": [...]
}
```

## Run the API

```bash
uv run fastapi dev main.py
```

Then open:

```text
http://127.0.0.1:8000/docs
```
