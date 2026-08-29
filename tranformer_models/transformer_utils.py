import torch


ID2LABEL = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise"
}


def predict_emotions(
    model,
    tokenizer,
    texts,
    device
):
    model.eval()

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    predicted_ids = torch.argmax(
        probabilities,
        dim=1
    )

    results = []

    for predicted_id, probability_vector in zip(
        predicted_ids,
        probabilities
    ):
        predicted_id = predicted_id.item()

        emotion = ID2LABEL[predicted_id]

        confidence = probability_vector[
            predicted_id
        ].item()

        results.append(
            (emotion, confidence)
        )

    return results