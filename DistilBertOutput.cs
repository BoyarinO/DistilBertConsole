using Microsoft.ML.Data;

public class DistilBertOutput
{
    [ColumnName("logits")]
    public float[] Logits { get; set; }
}