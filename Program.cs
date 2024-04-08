
class Program
{
    static void Main(string[] args)
    {   
        // Init variables
        var testText = "As of November 30 , 2015 , $ 151.8 million of the originated loans were sold into a securitization trust but not settled and thus were included as receivables , net . Notes and Other Debts Payable In November 2013 , the Rialto segment originally issued $ 250 million aggregate principal amount of the 7.00 % senior notes due 2018 ( \" 7.00 % Senior Notes \" ) , at a price of 100 % in a private placement .";
        int maxSequenceLength = 200;
        var modelPath = "/Users/olesboiaryn/Downloads/distilbert-base-uncased_ner_finetuned_onnx/model.onnx";
        var vocabPath = "/Users/olesboiaryn/Downloads/distilbert_tokenizer-vocab.txt";
        var labelsPath = "/Users/olesboiaryn/Downloads/distilbert-base-uncased_ner_finetuned_onnx/labels_finer-139_top_rows_10000_top_labels_4.txt";

        var distilBertConfig = new DistilBertConfig(maxSequenceLength, modelPath, vocabPath, labelsPath);
        var distilBert = new DistilBert(distilBertConfig);

        // Process text & predict labels
        (var sequenceLabels, var sequnceTokens) = distilBert.Predict(testText);

        // Print result
        Console.WriteLine("Predicted Labels:");
        var labelCounter = 0;
        foreach (var label in sequenceLabels)
        {
            var labelToken = sequnceTokens[labelCounter];
            Console.WriteLine(labelToken + " --- " + label);
            
            labelCounter++;
        }

        return;

    }
}
