public class DistilBertConfig
    {
        public DistilBertConfig(int maxSequenceLength, string modelPath, string vocabPath, string labelsPath)
        {
            this.MaxSequenceLength = maxSequenceLength;
            this.ModelPath = modelPath;
            this.VocabPath = vocabPath;
            this.LabelsPath = labelsPath;
        }

        public int MaxSequenceLength { get; set; }

        public string ModelPath { get; set; }

        public string VocabPath { get; set; }

        public string LabelsPath { get; set; }

    }