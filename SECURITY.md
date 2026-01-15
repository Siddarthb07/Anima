# Security

## Reporting

If you discover a security vulnerability in **this repository’s code** (e.g., API injection, unsafe deserialization in shipped paths), please report it responsibly:

- Open a **private** security advisory on the hosting platform if available, **or**
- Contact the maintainers with a clear reproducer and impact statement.

Do **not** file public issues with exploit details until a fix is agreed.

## Deployment notes

This project runs a **local FastAPI server** and serves a **dashboard**. Treat it like any other ML web stack:

- Do **not** expose unauthenticated instances to the public internet without your own auth, rate limits, and threat modeling.
- **Do not** commit `.env` files with API keys (`HF_TOKEN`, etc.).
- Hugging Face **model execution** is arbitrary code loading **trusted weights** only from sources **you** trust.

## Scope

We cannot patch vulnerabilities in **upstream** PyTorch, transformers, or OS kernels here — upgrade dependencies per your own policy.
