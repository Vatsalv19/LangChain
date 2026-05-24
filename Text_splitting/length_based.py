from langchain.text_splitter import CharacterTextSplitter

text = """
These are the tests for the length-based text splitter.
we will test the following cases:
1. Basic functionality: Test that the text is split correctly based on the specified length.
2. Edge cases: Test how the splitter handles edge cases, such as when the text is shorter than the specified length or when the text is exactly the specified length.
3. Special characters: Test how the splitter handles special characters, such as newlines, tabs

"""
splitter  = CharacterTextSplitter(
    separator = '',
    chunk_size = 100,
    chunk_overlap = 0

)
result = splitter.split_text(text)
print(result)