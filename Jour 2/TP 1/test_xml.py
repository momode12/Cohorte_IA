import xml.dom.minidom

dom = xml.dom.minidom.parse("data.xml")

xml_propre = dom.toprettyxml(indent="  ")
print(xml_propre)

racine = dom.documentElement
livres = racine.getElementsByTagName("livre")

for livre in livres:
    cat = livre.getAttribute("categorie")
    print(f"--- Catégorie : {cat} ---")
    
    titre_element = livre.getElementsByTagName("titre")[0]
    auteur_element = livre.getElementsByTagName("auteur")[0]
    
    print(f"Titre : {titre_element.firstChild.data}")
    print(f"Auteur : {auteur_element.firstChild.data}")
    print()