CREATE DATABASE IF NOT EXISTS GestionVentes;
USE GestionVentes;

CREATE TABLE Clients (
    ClientID INT AUTO_INCREMENT PRIMARY KEY,
    Nom VARCHAR(50),
    Prenom VARCHAR(50),
    Adresse VARCHAR(100),
    Email VARCHAR(100),
    NumeroTelephone VARCHAR(15)
);

CREATE TABLE Employes (
    EmployeID INT AUTO_INCREMENT PRIMARY KEY,
    Nom VARCHAR(50),
    Prenom VARCHAR(50),
    Fonction VARCHAR(50),
    Email VARCHAR(100),
    NumeroTelephone VARCHAR(15)
);

CREATE TABLE Fournisseurs (
    FournisseurID INT AUTO_INCREMENT PRIMARY KEY,
    NomFournisseur VARCHAR(50),
    Adresse VARCHAR(100),
    Email VARCHAR(100),
    NumeroTelephone VARCHAR(15)
);


CREATE TABLE Produits (
    ProduitID INT AUTO_INCREMENT PRIMARY KEY,
    NomProduit VARCHAR(50),
    Description TEXT,
    PrixUnitaire DECIMAL(10,2),
    FournisseurID INT,
    FOREIGN KEY (FournisseurID) REFERENCES Fournisseurs(FournisseurID)
);

CREATE TABLE Ventes (
    VenteID INT AUTO_INCREMENT PRIMARY KEY,
    DateVente DATE,
    ClientID INT,
    EmployeID INT,
    ProduitID INT,
    QuantiteVendue INT,
    MontantTotal DECIMAL(10,2),
    FOREIGN KEY (ClientID) REFERENCES Clients(ClientID),
    FOREIGN KEY (EmployeID) REFERENCES Employes(EmployeID),
    FOREIGN KEY (ProduitID) REFERENCES Produits(ProduitID)
);