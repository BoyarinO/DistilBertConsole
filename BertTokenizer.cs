
///ref: https://github.com/codito/semanticsearch
/// <summary>
/// Word piece tokenizer based on bert-base-uncased model in transformers.
/// See tools/tokenizer.py for examples in python.
/// </summary>
public class BertTokenizer
{
    public readonly List<string> vocabulary;

    public BertTokenizer(List<string> vocabulary)
    {
        this.vocabulary = vocabulary;
    }

    /// <summary>
    /// Tokenize a set of strings.
    /// </summary>
    /// <param name="texts">List of strings.</param>
    /// <returns>List of tokens.</returns>
    public List<(string Token, int VocabularyIndex)> Tokenize(IEnumerable<string> texts)
    {
        // [CLS] Words of sentence [SEP] Words of next sentence [SEP]
        IEnumerable<string> tokens = new[]
            {
                DefaultTokens.Classification
            };

        foreach (var text in texts)
        {
            tokens = tokens.Concat(this.TokenizeSentence(text));
            tokens = tokens.Concat(new[] { DefaultTokens.Separation });
        }

        return tokens
            .SelectMany(this.TokenizeSubwords)
            .ToList();
    }

    /**
        * Some words in the vocabulary are too big and will be broken up in to subwords
        * Example "Embeddings"
        * [‘em’, ‘##bed’, ‘##ding’, ‘##s’]
        * https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
        * https://developpaper.com/bert-visual-learning-of-the-strongest-nlp-model/
        * https://medium.com/@_init_/why-bert-has-3-embedding-layers-and-their-implementation-details-9c261108e28a
        */
    private IEnumerable<(string Token, int VocabularyIndex)>
        TokenizeSubwords(string word)
    {
        if (this.vocabulary.Contains(word))
        {
            return new (string, int)[]
            {
                (word, this.vocabulary.IndexOf(word))
            };
        }

        var tokens = new List<(string, int)>();
        var remaining = word;

        while (!string.IsNullOrEmpty(remaining) && remaining.Length > 2)
        {
            var prefix = this.vocabulary.Where(remaining.StartsWith)
                .OrderByDescending(o => o.Length)
                .FirstOrDefault();

            if (prefix == null)
            {
                tokens.Add((DefaultTokens.Unknown,
                    this.vocabulary.IndexOf(DefaultTokens.Unknown)));

                return tokens;
            }

            var replaced = remaining.Replace(prefix, "##");
            if (replaced.Length == remaining.Length)
            {
                break;
            }

            remaining = replaced;

            tokens.Add((prefix, this.vocabulary.IndexOf(prefix)));
        }

        if (!string.IsNullOrWhiteSpace(word) && !tokens.Any())
        {
            tokens.Add((DefaultTokens.Unknown,
                this.vocabulary.IndexOf(DefaultTokens.Unknown)));
        }

        return tokens;
    }


    private IEnumerable<string> TokenizeSentence(string text)
    {
        // remove spaces and split the , . : ; etc..
        return text.Split(
            new[] { " ", "   ", "\r\n" },
            StringSplitOptions.None)
            .SelectMany(o =>
                    o.SplitAndKeeps(".,;:\\/?!#$%()=+-*\"'–_`<>&^@{}[]|~'"
                    .ToArray()))
            .Select(o => o.ToLower());
    }

    public List<int> DefaultTokensIndices{
        get {return new List<int>(){
            this.vocabulary.IndexOf(DefaultTokens.Padding),
            this.vocabulary.IndexOf(DefaultTokens.Unknown),
            this.vocabulary.IndexOf(DefaultTokens.Classification),
            this.vocabulary.IndexOf(DefaultTokens.Separation),
            this.vocabulary.IndexOf(DefaultTokens.Mask)
        };}
    }

    public class DefaultTokens
    {
        public const string Padding = "[PAD]";
        public const string Unknown = "[UNK]";
        public const string Classification = "[CLS]";
        public const string Separation = "[SEP]";
        public const string Mask = "[MASK]";
    }
}


public static class LinqExtension
{
public static IEnumerable<string> SplitAndKeeps(this string s, char[] delims)
    {
        int start = 0, index;

        while ((index = s.IndexOfAny(delims, start)) != -1)
        {
            if(index-start > 0)
                yield return s.Substring(start, index - start);
            yield return s.Substring(index, 1);
            start = index + 1;
        }

        if (start < s.Length)
        {
            yield return s.Substring(start);
        }
    }
}