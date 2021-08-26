import wikipediaapi

def wiki_ko(query):

    wiki=wikipediaapi.Wikipedia('ko', extract_format=wikipediaapi.ExtractFormat.WIKI)

    page_py = wiki.page(f'{query}')
    print(f"Page - {query} - Exists: {page_py.exists()}")
    print(f"Page - Title: {page_py.title}")

    p_wiki = wiki.page(f"{query}")
    print(p_wiki.text)

    with open(f"{query}_wiki.txt", "w", encoding='utf-8') as f:
        f.write(p_wiki.text)

def wiki_en(query):
    
    wiki=wikipediaapi.Wikipedia('en', extract_format=wikipediaapi.ExtractFormat.WIKI)

    page_py = wiki.page(f'{query}')
    print(f"Page - {query} - Exists: {page_py.exists()}")
    print(f"Page - Title: {page_py.title}")

    p_wiki = wiki.page(f"{query}")
    print(p_wiki.text)

    with open(f"{query}_wiki.txt", "w", encoding='utf-8') as f:
        f.write(p_wiki.text)



print(wiki_ko('에스파'))
# print(wiki_en('python'))