# DistilBert ONNX Inference Application

This .NET Console Application demonstrates how to use a trained DistilBert model in ONNX format for Named Entity Recognition (NER) tasks. The application leverages the Microsoft ML.NET framework to load the ONNX model, preprocess text inputs, perform inference, and decode the predictions to human-readable labels.


## Dependencies

- .NET 8.0 SDK
- `Microsoft.ML` version `3.0.1`
- `Microsoft.ML.OnnxRuntime` version `1.17.1`
- `Microsoft.ML.OnnxTransformer` version `3.0.1`

## Configuration & Run

Before running the application, configure the following paths in `Program.cs` to point to your model, tokenizer vocabulary, and labels file:

```csharp
var filesDir = "distilbert-base-uncased_ner_finetuned_onnx";
var modelPath = Path.Combine(Environment.CurrentDirectory, filesDir, "model.onnx");
var vocabPath = Path.Combine(Environment.CurrentDirectory, filesDir, "distilbert_tokenizer-vocab.txt");
var labelsPath = Path.Combine(Environment.CurrentDirectory, filesDir, "labels_finer-139_top_rows_10000_top_labels_4.txt");
```
and pass your custom text via 
```chsarp
var testText = "As of November 30 , 2015 , $ 151.8 million of the originated loans were sold into a securitization trust but not settled and thus were included as receivables , net . Notes and Other Debts Payable In November 2013 , the Rialto segment originally issued $ 250 million aggregate principal amount of the 7.00 % senior notes due 2018 ( \" 7.00 % Senior Notes \" ) , at a price of 100 % in a private placement .";
```

By running `Program.cs` you will get the following output `[token] --- [label]`:
```
Predicted Labels:
other --- O
debts --- O
pay --- O
##able --- O
in --- O
november --- O
2013 --- O
, --- O
the --- O
segment --- O
originally --- O
issued --- O
$ --- O
250 --- B-DebtInstrumentFaceAmount
```


## References
 - As a reference for DistilBertTokenizer was used [DistilBERT Sematic search project](https://github.com/codito/semanticsearch).
