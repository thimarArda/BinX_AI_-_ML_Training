# Banking Chatbot with DistilBERT

## Overview

This project uses a **Transformer model** to understand text related to banking services and identify what the user is asking about. The goal is to build the basic text-understanding part of a chatbot that can receive a user's message, classify its intent, and provide an appropriate response.

Instead of building a Transformer from scratch, I use the pre-trained **DistilBERT** model and fine-tune it for the banking-related task.

## Model

The Transformer model used in this project is **DistilBERT (`distilbert-base-uncased`)**.

DistilBERT is a smaller and faster version of BERT. It is already pre-trained on a large amount of text, so it has learned general language patterns. I then fine-tune it so that it can understand different types of banking-related questions.

## Dataset

The dataset used is **Banking77**.

Banking77 contains customer questions related to banking and divides them into **77 different intents**. These intents represent different things a customer might want help with, such as card problems, cash withdrawals, payments, fees, account issues, and other banking services.

## How the Project Works

The basic process is:

```text
User Input
    ↓
Tokenization
    ↓
DistilBERT
    ↓
Intent Classification
    ↓
Appropriate Response
```

The user enters a message about a banking service. The text is converted into tokens and passed to DistilBERT. The model then predicts the intent of the message from the 77 categories in the dataset. Based on the predicted intent, the chatbot can choose an appropriate response.

### Example

For example, the user might enter:

> "Why was I charged an extra fee for my card payment?"

The model processes the text and predicts an intent such as:

```text
Intent: card_payment_fee_charged
```

The chatbot can then respond with something related to card payment fees:

> "It looks like you are asking about a fee charged for a card payment. I can help you understand why this fee may have been applied."

The Transformer is therefore responsible for **understanding and classifying the user's message**, while the response system uses that classification to decide how to answer.

## Project Goal

The main goal of this project is to understand how **Transformers and attention mechanisms can be used for text understanding** and to build a simple chatbot foundation using a real-world text dataset.

This project can also be extended later into a more complete chatbot application by adding more responses, a user interface, and a more advanced response-generation system.