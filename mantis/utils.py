def create_prodigy_spans(doc):
    pred = []
    for entity in doc.ents:
        pred.append({"label": entity.label_, "start": entity.start, "end": entity.end})
    return pred
