import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from pprint import pprint

def scrape_books(url):
    livres = []
    while url:
        soup = BeautifulSoup(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).content, "html.parser")
        for livre in soup.find_all("article", class_="product_pod"):
            titre = livre.find("h3").find("a")["title"]
            prix = livre.find("p", class_="price_color").text
            livres.append({"titre": titre, "prix": prix})
        next_btn = soup.find("li", class_="next")
        url = url.rsplit("/", 1)[0] + "/" + next_btn.find("a")["href"] if next_btn else None
        time.sleep(0.3)
    return livres

print("Scraping...")
data = scrape_books("https://books.toscrape.com/catalogue/category/books_1/index.html")

print(f"\n DONNÉES BRUTES ({len(data)} livres):")
for i, livre in enumerate(data, 1):
    print(f"  {i}. {livre}")

print("\n Aperçu des 5 premiers:")
for i, b in enumerate(data[:5], 1):
    print(f"  {i}. {b['titre'][:35]}... - {b['prix']}")

print("\n Création du DataFrame...")
df = pd.DataFrame(data)
print(df.head(10))
print(f"\nShape: {df.shape}")
print(f"Colonnes: {df.columns.tolist()}")

df.to_csv("books.csv", index=False, encoding="utf-8")
print("\n Sauvegardé dans books.csv")