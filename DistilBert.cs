using Microsoft.ML;

/// <summary>
/// The DistilBert class encapsulates the configuration, tokenizer, and label information
/// for interacting with a DistilBERT model. It provides functionality to prepare input
/// data for the model and handle the encoding of text for NER tasks.
/// </summary>
public class DistilBert
{
    // Names of the output columns from the model
    public static readonly string[] OutputColumnNames =
    {
        "logits"
    };

    // Names of the input columns for the model
    public static readonly string[] InputColumnNames =
    {
        "input_ids", "attention_mask"
    };

    // Configuration for the DistilBert instance
    private readonly DistilBertConfig config;

    // Tokenizer for converting text into tokens that can be fed into the DistilBERT model
    public readonly BertTokenizer tokenizer;

    // List of labels that the model can predict
    public readonly List<string> labels;

    // Property to get the count of labels
    public int LabelsCount {
        get { return this.labels.Count;}
    }

    /// <summary>
    /// Constructor for the DistilBert class. Initializes the tokenizer, label list, and configuration.
    /// </summary>
    /// <param name="config">Configuration settings for DistilBERT model</param>
    public DistilBert(DistilBertConfig config)
    {
        this.config = config;
        // Initialize the tokenizer with the vocabulary from a file
        this.tokenizer = new BertTokenizer(File.ReadAllLines(config.VocabPath).ToList());
        
        // Initialize the labels that the model is capable of predicting
        this.labels = File.ReadAllLines(config.LabelsPath).ToList();
    }


    /// <summary>
    /// Prepares the input for the DistilBert and predicts NER labels for the provided text
    /// </summary>
    /// <param name="inputText">Text to predict</param>
    /// <returns>Tuple of lists: Sequence labels, Sequence tokens</returns>
    public (List<string>, List<string>)  Predict(string inputText)
    {
        var mlContext = new MLContext();

        var encodeData = PrepareInput(inputText);
        var batchSize = 1;
        var inputShape = InputColumnNames.ToDictionary(item => item, item => new[] { batchSize, config.MaxSequenceLength });
        
        // Define data processing pipeline
        var pipeline = mlContext.Transforms.ApplyOnnxModel(
            OutputColumnNames, 
            InputColumnNames,
            config.ModelPath,
            inputShape,
            null,
            true
           );

        var inputData = new List<DistilBertInput>(){encodeData};
        var dataView = mlContext.Data.LoadFromEnumerable(inputData);
        var model = pipeline.Fit(dataView);


        // Create prediction engine
        var predictionEngine = mlContext.Model.CreatePredictionEngine<DistilBertInput, DistilBertOutput>(
            model
            );
        
        // Make prediction
        var prediction = predictionEngine.Predict(encodeData); 
        var logits = prediction.Logits.ToList();

        // Decode logits
        var iteration = 0;
        var numIterations = logits.Count() / LabelsCount;
        var sequenceLabels = new List<string>();
        var sequnceTokens = new List<string>();

        while(iteration < numIterations){
            var token = (int)encodeData.InputIds[iteration];

            // Skip technical tokens 
            if (!tokenizer.DefaultTokensIndices.Contains(token)){
                var startIndex = iteration*LabelsCount;
                var endIndex = (iteration+1)*LabelsCount;
                var logitsSlice = logits[startIndex..endIndex];
                var maxIndex = logitsSlice.IndexOf(logitsSlice.Max());
                var className = labels[maxIndex];
                
                sequenceLabels.Add(className);
                sequnceTokens.Add(tokenizer.vocabulary[token]);
            }
            
            iteration++;
        }

        return (sequenceLabels, sequnceTokens);
    }


    /// <summary>
    /// Prepares the input for the DistilBert model by encoding the provided text.
    /// </summary>
    /// <param name="text">Text to encode</param>
    /// <returns>DistilBertInput object containing encoded input data</returns>
    private DistilBertInput PrepareInput(string text)
    {
        // Tokenize the text and encode it for the model input
        return this.Encode(this.tokenizer.Tokenize(new[] { text }), this.config.MaxSequenceLength);
    }


    /// <summary>
    /// Encodes tokenized text into model input format, applying padding as needed.
    /// </summary>
    /// <param name="tokens">Tokenized text</param>
    /// <param name="maxSequenceLength">Maximum sequence length for the model input</param>
    /// <returns>Encoded input suitable for the DistilBert model</returns>
    private DistilBertInput Encode(
        List<(string Token, int Index)> tokens,
        int maxSequenceLength)
    {
        // Create padding based on the max sequence length and the number of tokens
        var padding = Enumerable
            .Repeat(0L, maxSequenceLength - tokens.Count)
            .ToList();

        // Convert token indexes to long and apply padding
        var tokenIndexes = tokens
            .Select(token => (long)token.Index)
            .Concat(padding)
            .ToArray();

        // Generate and pad segment indexes (for models that use them, not used in DistilBERT)
        var segmentIndexes = this.GetSegmentIndexes(tokens)
            .Concat(padding)
            .ToArray();

        // Create attention mask with padding
        var inputMask =
            tokens.Select(o => 1L)
                .Concat(padding)
                .ToArray();

        // Return the model input structure
        return new DistilBertInput
        {
            InputIds = tokenIndexes,
            AttentionMask = inputMask
        };
    }

    /// <summary>
    /// Generates segment indexes for tokens. 
    /// </summary>
    /// <param name="tokens">Tokenized text</param>
    /// <returns>A sequence of segment indexes for each token</returns>
    private IEnumerable<long> GetSegmentIndexes(
        List<(string Token, int Index)> tokens)
    {
        var segmentIndex = 0;
        var segmentIndexes = new List<long>();

        // Assign segment indexes, incrementing for each separation token
        foreach (var (token, _) in tokens)
        {
            segmentIndexes.Add(segmentIndex);

            if (token == BertTokenizer.DefaultTokens.Separation)
            {
                segmentIndex++;
            }
        }

        return segmentIndexes;
    }
}
