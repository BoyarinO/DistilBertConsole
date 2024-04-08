using Microsoft.ML.Data;

public class DistilBertInput
{
    // Dimensions: batch, sequence
    [VectorType(1,200)]
    [ColumnName("input_ids")]
    public long[] InputIds { get; set; }

    // Dimensions: batch, sequence
    [VectorType(1,200)]
    [ColumnName("attention_mask")]
    public long[] AttentionMask { get; set; }
}