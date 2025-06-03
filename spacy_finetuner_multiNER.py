import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import json
from pathlib import Path
import argparse
from typing import List, Tuple, Dict, Optional
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
from io import StringIO


class SpacyMultiEntityFineTuner:
    """Classe pour faciliter le fine-tuning de modèles spaCy pour la NER multi-entités"""
    
    def __init__(self, model_size: str = "sm", lang: str = "fr"):
        """
        Initialise le fine-tuner
        
        Args:
            model_size: Taille du modèle ("sm", "md", "lg")
            lang: Langue du modèle (par défaut "fr" pour français)
        """
        self.model_size = model_size
        self.lang = lang
        self.model_name = f"{lang}_core_news_{model_size}"
        self.nlp = None
        self.training_data = []
        
        # Dictionnaires pour stocker les différents types d'entités
        self.entities = {
            'PER': [],  # Personnes
            'ORG': [],  # Organisations
            'LOC': []   # Lieux
        }
        
    def load_base_model(self):
        """Charge le modèle spaCy de base"""
        try:
            self.nlp = spacy.load(self.model_name)
            print(f"✓ Modèle {self.model_name} chargé avec succès")
        except OSError:
            print(f"⚠ Le modèle {self.model_name} n'est pas installé.")
            print(f"Installez-le avec: python -m spacy download {self.model_name}")
            raise
            
        # Ajouter le pipeline NER s'il n'existe pas
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")
            
        return ner
    
    def load_entities_from_file(self, filepath: str, entity_type: str) -> List[str]:
        """
        Charge les entités depuis un fichier
        
        Args:
            filepath: Chemin vers le fichier d'entités (une entité par ligne)
            entity_type: Type d'entité ('PER', 'ORG', 'LOC')
        
        Returns:
            Liste des entités
        """
        if not filepath:
            return []
            
        entities = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                entity = line.strip()
                if entity:
                    entities.append(entity)
        
        self.entities[entity_type] = entities
        
        entity_name = {'PER': 'personnes', 'ORG': 'organisations', 'LOC': 'lieux'}[entity_type]
        print(f"✓ {len(entities)} {entity_name} chargé(e)s")
        return entities
    
    def load_sentence_templates(self, filepath: str) -> List[str]:
        """
        Charge les phrases types depuis un fichier
        
        Format supporté:
        - {NOM} pour les personnes
        - {ORG} pour les organisations
        - {LIEU} pour les lieux
        
        Args:
            filepath: Chemin vers le fichier de phrases types
        
        Returns:
            Liste des phrases types
        """
        templates = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                template = line.strip()
                if template:
                    templates.append(template)
        
        print(f"✓ {len(templates)} phrases types chargées")
        return templates
    
    def generate_training_data(self, templates: List[str], 
                             samples_per_template: int = 5) -> List[Tuple[str, Dict]]:
        """
        Génère des données d'entraînement en combinant entités et phrases types
        
        Args:
            templates: Liste des phrases types avec placeholders
            samples_per_template: Nombre d'échantillons à générer par template
        
        Returns:
            Liste de tuples (texte, annotations)
        """
        training_examples = []
        
        for template in templates:
            # Identifier les placeholders dans le template
            placeholders_info = []
            
            # Chercher tous les placeholders avec leur type
            for match in re.finditer(r'\{(NOM|ORG|LIEU)\}', template):
                placeholder_type = match.group(1)
                entity_type = {'NOM': 'PER', 'ORG': 'ORG', 'LIEU': 'LOC'}[placeholder_type]
                placeholders_info.append({
                    'start': match.start(),
                    'end': match.end(),
                    'type': entity_type,
                    'placeholder': match.group(0)
                })
            
            if not placeholders_info:
                continue
            
            # Générer plusieurs exemples pour chaque template
            for _ in range(samples_per_template):
                text = template
                entities = []
                offset = 0
                
                # Remplacer chaque placeholder
                for ph_info in placeholders_info:
                    entity_type = ph_info['type']
                    
                    # Vérifier qu'il y a des entités disponibles pour ce type
                    if not self.entities[entity_type]:
                        continue
                    
                    # Sélectionner aléatoirement une entité
                    entity = random.choice(self.entities[entity_type])
                    
                    # Calculer la position avec l'offset
                    placeholder_pos = text.find(ph_info['placeholder'], offset)
                    if placeholder_pos == -1:
                        continue
                    
                    # Remplacer le placeholder
                    text = (text[:placeholder_pos] + entity + 
                           text[placeholder_pos + len(ph_info['placeholder']):])
                    
                    # Ajouter l'annotation
                    start = placeholder_pos
                    end = placeholder_pos + len(entity)
                    entities.append((start, end, entity_type))
                    
                    # Mettre à jour l'offset
                    offset = end
                
                # Ajouter l'exemple aux données d'entraînement
                if entities:  # S'assurer qu'au moins une entité a été ajoutée
                    training_examples.append((text, {"entities": entities}))
        
        print(f"✓ {len(training_examples)} exemples d'entraînement générés")
        return training_examples
    
    def prepare_training_data(self, examples: List[Tuple[str, Dict]]) -> List[Example]:
        """
        Prépare les données pour l'entraînement spaCy
        
        Args:
            examples: Liste de tuples (texte, annotations)
        
        Returns:
            Liste d'objets Example pour spaCy
        """
        training_examples = []
        
        for text, annotations in examples:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            training_examples.append(example)
            
        return training_examples
    
    def train(self, training_examples: List[Example], n_iter: int = 30, 
              drop: float = 0.35, batch_size: int = 8):
        """
        Entraîne le modèle NER
        
        Args:
            training_examples: Données d'entraînement
            n_iter: Nombre d'itérations
            drop: Taux de dropout
            batch_size: Taille des batches
        """
        # Obtenir le pipeline NER
        ner = self.nlp.get_pipe("ner")
        
        # Ajouter tous les labels utilisés
        for _, annotations in self.training_data:
            for ent in annotations.get("entities", []):
                ner.add_label(ent[2])
        
        # Désactiver les autres pipelines pendant l'entraînement
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            
            print("\n🚀 Début de l'entraînement...")
            
            for iteration in range(n_iter):
                print(f"\nItération {iteration + 1}/{n_iter}")
                
                # Mélanger les données
                random.shuffle(training_examples)
                losses = {}
                
                # Créer les batches
                batches = minibatch(training_examples, size=compounding(4.0, batch_size, 1.001))
                
                for batch in batches:
                    # Mettre à jour le modèle
                    self.nlp.update(batch, sgd=optimizer, drop=drop, losses=losses)
                
                print(f"Perte NER: {losses.get('ner', 0):.4f}")
        
        print("\n✓ Entraînement terminé!")
    
    def evaluate(self, test_examples: List[Tuple[str, Dict]]) -> Dict:
        """
        Évalue le modèle sur des données de test
        
        Args:
            test_examples: Exemples de test
        
        Returns:
            Métriques d'évaluation
        """
        scorer = self.nlp.evaluate(self.prepare_training_data(test_examples))
        return scorer
    
    def save_model(self, output_dir: str):
        """
        Sauvegarde le modèle fine-tuné
        
        Args:
            output_dir: Répertoire de sortie
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Mettre à jour les métadonnées du modèle
        self.nlp.meta["name"] = f"custom_multi_ner_{self.model_size}"
        self.nlp.meta["version"] = "1.0.0"
        self.nlp.meta["description"] = "Modèle spaCy fine-tuné pour la reconnaissance multi-entités (PER, ORG, LOC)"
        self.nlp.meta["author"] = "SpacyMultiEntityFineTuner"
        self.nlp.meta["lang"] = self.lang
        self.nlp.meta["pipeline"] = list(self.nlp.pipe_names)
        self.nlp.meta["labels"] = {
            "ner": list(self.nlp.get_pipe("ner").labels)
        }
        
        # Sauvegarder le modèle
        self.nlp.to_disk(output_path)
        print(f"\n✓ Modèle sauvegardé dans: {output_path}")
        
        # Créer le fichier config si nécessaire
        config_path = output_path / "config.cfg"
        if not config_path.exists():
            self.nlp.config.to_disk(config_path)
    
    def test_model(self, test_texts: List[str]):
        """
        Teste le modèle sur quelques exemples
        
        Args:
            test_texts: Textes à tester
        """
        print("\n🧪 Test du modèle:")
        print("-" * 50)
        
        for text in test_texts:
            doc = self.nlp(text)
            print(f"\nTexte: {text}")
            print("Entités détectées:")
            for ent in doc.ents:
                print(f"  - {ent.text} ({ent.label_})")


class SpacyMultiEntityFineTunerGUI:
    """Interface graphique pour le fine-tuning de modèles spaCy multi-entités"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SpaCy Fine-Tuner - Entraînement NER Multi-Entités")
        self.root.geometry("900x700")
        
        # Variables
        self.names_file = tk.StringVar()
        self.orgs_file = tk.StringVar()
        self.places_file = tk.StringVar()
        self.templates_file = tk.StringVar()
        self.output_dir = tk.StringVar(value="model_multi_ner")
        self.model_size = tk.StringVar(value="sm")
        self.iterations = tk.IntVar(value=30)
        self.samples_per_template = tk.IntVar(value=5)
        self.is_training = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Frame principal avec scrollbar
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Frame principal
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Titre
        title = ttk.Label(main_frame, text="Fine-Tuning SpaCy - Reconnaissance Multi-Entités", 
                         font=('Helvetica', 16, 'bold'))
        title.pack(pady=10)
        
        # Section fichiers d'entités
        entities_frame = ttk.LabelFrame(main_frame, text="Fichiers d'entités", padding="10")
        entities_frame.pack(fill="x", pady=10, padx=5)
        
        # Personnes
        person_frame = ttk.Frame(entities_frame)
        person_frame.pack(fill="x", pady=5)
        ttk.Label(person_frame, text="👤 Personnes (PER):", width=20).pack(side="left", padx=5)
        ttk.Entry(person_frame, textvariable=self.names_file, width=40).pack(side="left", padx=5)
        ttk.Button(person_frame, text="Parcourir", 
                  command=lambda: self.browse_file(self.names_file)).pack(side="left", padx=5)
        
        # Organisations
        org_frame = ttk.Frame(entities_frame)
        org_frame.pack(fill="x", pady=5)
        ttk.Label(org_frame, text="🏢 Organisations (ORG):", width=20).pack(side="left", padx=5)
        ttk.Entry(org_frame, textvariable=self.orgs_file, width=40).pack(side="left", padx=5)
        ttk.Button(org_frame, text="Parcourir", 
                  command=lambda: self.browse_file(self.orgs_file)).pack(side="left", padx=5)
        
        # Lieux
        place_frame = ttk.Frame(entities_frame)
        place_frame.pack(fill="x", pady=5)
        ttk.Label(place_frame, text="📍 Lieux (LOC):", width=20).pack(side="left", padx=5)
        ttk.Entry(place_frame, textvariable=self.places_file, width=40).pack(side="left", padx=5)
        ttk.Button(place_frame, text="Parcourir", 
                  command=lambda: self.browse_file(self.places_file)).pack(side="left", padx=5)
        
        # Section phrases types
        templates_frame = ttk.LabelFrame(main_frame, text="Phrases types", padding="10")
        templates_frame.pack(fill="x", pady=10, padx=5)
        
        template_info = ttk.Label(templates_frame, 
                                 text="Utilisez {NOM} pour les personnes, {ORG} pour les organisations, {LIEU} pour les lieux",
                                 font=('Helvetica', 9, 'italic'))
        template_info.pack(pady=5)
        
        template_file_frame = ttk.Frame(templates_frame)
        template_file_frame.pack(fill="x")
        ttk.Label(template_file_frame, text="Fichier de phrases:", width=20).pack(side="left", padx=5)
        ttk.Entry(template_file_frame, textvariable=self.templates_file, width=40).pack(side="left", padx=5)
        ttk.Button(template_file_frame, text="Parcourir", 
                  command=lambda: self.browse_file(self.templates_file)).pack(side="left", padx=5)
        
        # Section paramètres
        params_frame = ttk.LabelFrame(main_frame, text="Paramètres", padding="10")
        params_frame.pack(fill="x", pady=10, padx=5)
        
        # Première ligne de paramètres
        params_row1 = ttk.Frame(params_frame)
        params_row1.pack(fill="x", pady=5)
        
        ttk.Label(params_row1, text="Taille du modèle:").pack(side="left", padx=5)
        model_combo = ttk.Combobox(params_row1, textvariable=self.model_size, 
                                  values=["sm", "md", "lg"], state="readonly", width=10)
        model_combo.pack(side="left", padx=5)
        
        ttk.Label(params_row1, text="Itérations:").pack(side="left", padx=20)
        ttk.Spinbox(params_row1, from_=10, to=100, textvariable=self.iterations, 
                   width=10).pack(side="left", padx=5)
        
        # Deuxième ligne de paramètres
        params_row2 = ttk.Frame(params_frame)
        params_row2.pack(fill="x", pady=5)
        
        ttk.Label(params_row2, text="Échantillons/template:").pack(side="left", padx=5)
        ttk.Spinbox(params_row2, from_=1, to=20, textvariable=self.samples_per_template, 
                   width=10).pack(side="left", padx=5)
        
        ttk.Label(params_row2, text="Dossier de sortie:").pack(side="left", padx=20)
        ttk.Entry(params_row2, textvariable=self.output_dir, width=25).pack(side="left", padx=5)
        
        # Boutons d'action
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(pady=10)
        
        self.train_button = ttk.Button(action_frame, text="🚀 Lancer l'entraînement", 
                                      command=self.start_training, style='Accent.TButton')
        self.train_button.pack(side="left", padx=5)
        
        ttk.Button(action_frame, text="🧪 Tester le modèle", 
                  command=self.test_model).pack(side="left", padx=5)
        
        ttk.Button(action_frame, text="📋 Générer des exemples", 
                  command=self.generate_example_files).pack(side="left", padx=5)
        
        # Zone de log
        log_frame = ttk.LabelFrame(main_frame, text="Journal d'exécution", padding="10")
        log_frame.pack(fill="both", expand=True, pady=10, padx=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.log_text.pack(fill="both", expand=True)
        
        # Barre de progression
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill="x", pady=5, padx=5)
        
        # Style
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Helvetica', 10, 'bold'))
        
    def browse_file(self, var):
        """Ouvre un dialogue pour sélectionner un fichier"""
        filename = filedialog.askopenfilename(
            title="Sélectionner un fichier",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        if filename:
            var.set(filename)
    
    def log(self, message):
        """Ajoute un message au journal"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def generate_example_files(self):
        """Génère des fichiers d'exemple pour aider l'utilisateur"""
        try:
            # Demander où sauvegarder les exemples
            directory = filedialog.askdirectory(title="Choisir le dossier pour les fichiers d'exemple")
            if not directory:
                return
            
            # Créer les fichiers d'exemple
            # Personnes
            with open(f"{directory}/exemples_personnes.txt", "w", encoding="utf-8") as f:
                f.write("Marie Dupont\n")
                f.write("Jean-Pierre Martin\n")
                f.write("Sophie Bernard\n")
                f.write("François Lefebvre\n")
                f.write("Catherine Rousseau\n")
                f.write("Michel Girard\n")
                f.write("Isabelle Moreau\n")
                f.write("Patrick Dubois\n")
                f.write("Nathalie Laurent\n")
                f.write("Thierry Petit\n")
            
            # Organisations
            with open(f"{directory}/exemples_organisations.txt", "w", encoding="utf-8") as f:
                f.write("Microsoft\n")
                f.write("Google France\n")
                f.write("Université de Paris\n")
                f.write("Hôpital Saint-Louis\n")
                f.write("Banque Nationale\n")
                f.write("Air France\n")
                f.write("Ministère de la Santé\n")
                f.write("Total Energies\n")
                f.write("Société Générale\n")
                f.write("CNRS\n")
            
            # Lieux
            with open(f"{directory}/exemples_lieux.txt", "w", encoding="utf-8") as f:
                f.write("Paris\n")
                f.write("Lyon\n")
                f.write("Marseille\n")
                f.write("Toulouse\n")
                f.write("Nice\n")
                f.write("Nantes\n")
                f.write("Strasbourg\n")
                f.write("Montpellier\n")
                f.write("Bordeaux\n")
                f.write("Lille\n")
            
            # Phrases types
            with open(f"{directory}/exemples_phrases_types.txt", "w", encoding="utf-8") as f:
                f.write("{NOM} travaille chez {ORG}.\n")
                f.write("Le rapport de {NOM} a été envoyé à {ORG}.\n")
                f.write("{NOM} et {NOM} se sont rencontrés à {LIEU}.\n")
                f.write("La conférence de {ORG} aura lieu à {LIEU}.\n")
                f.write("{NOM} est le directeur de {ORG} basé à {LIEU}.\n")
                f.write("J'ai rendez-vous avec {NOM} au bureau de {ORG}.\n")
                f.write("{ORG} a ouvert une nouvelle filiale à {LIEU}.\n")
                f.write("Le docteur {NOM} exerce à l'{ORG} de {LIEU}.\n")
                f.write("{NOM} a quitté {ORG} pour rejoindre {ORG}.\n")
                f.write("La réunion entre {NOM} et {NOM} est prévue à {LIEU}.\n")
            
            self.log(f"✅ Fichiers d'exemple créés dans: {directory}")
            messagebox.showinfo("Succès", f"Les fichiers d'exemple ont été créés dans:\n{directory}")
            
        except Exception as e:
            self.log(f"❌ Erreur lors de la création des fichiers: {str(e)}")
            messagebox.showerror("Erreur", f"Erreur lors de la création des fichiers:\n{str(e)}")
    
    def start_training(self):
        """Lance l'entraînement dans un thread séparé"""
        if self.is_training:
            messagebox.showwarning("Entraînement en cours", 
                                 "Un entraînement est déjà en cours!")
            return
            
        # Vérifier qu'au moins un fichier d'entités et le fichier de templates sont fournis
        if not self.templates_file.get():
            messagebox.showerror("Erreur", 
                               "Veuillez sélectionner au moins le fichier de phrases types!")
            return
        
        if not any([self.names_file.get(), self.orgs_file.get(), self.places_file.get()]):
            messagebox.showerror("Erreur", 
                               "Veuillez sélectionner au moins un fichier d'entités!")
            return
        
        # Lancer dans un thread
        self.is_training = True
        self.train_button.config(state='disabled')
        self.progress.start()
        
        thread = threading.Thread(target=self.run_training)
        thread.start()
    
    def run_training(self):
        """Exécute l'entraînement"""
        # Rediriger stdout vers le journal
        old_stdout = sys.stdout
        sys.stdout = TextRedirector(self.log)
        
        try:
            # Initialiser le fine-tuner
            self.log("Initialisation du fine-tuner multi-entités...")
            tuner = SpacyMultiEntityFineTuner(model_size=self.model_size.get(), lang="fr")
            
            # Charger le modèle de base
            self.log("Chargement du modèle de base...")
            tuner.load_base_model()
            
            # Charger les différents types d'entités
            self.log("\nChargement des entités...")
            if self.names_file.get():
                tuner.load_entities_from_file(self.names_file.get(), 'PER')
            if self.orgs_file.get():
                tuner.load_entities_from_file(self.orgs_file.get(), 'ORG')
            if self.places_file.get():
                tuner.load_entities_from_file(self.places_file.get(), 'LOC')
            
            # Charger les templates
            templates = tuner.load_sentence_templates(self.templates_file.get())
            
            # Générer les données d'entraînement
            self.log("\nGénération des données d'entraînement...")
            training_data = tuner.generate_training_data(
                templates, self.samples_per_template.get()
            )
            tuner.training_data = training_data
            
            # Diviser en ensemble d'entraînement et de test
            split_point = int(len(training_data) * 0.8)
            train_data = training_data[:split_point]
            test_data = training_data[split_point:]
            
            self.log(f"Données d'entraînement: {len(train_data)} exemples")
            self.log(f"Données de test: {len(test_data)} exemples")
            
            # Préparer les données d'entraînement
            self.log("\nPréparation des données...")
            training_examples = tuner.prepare_training_data(train_data)
            
            # Entraîner le modèle
            tuner.train(training_examples, n_iter=self.iterations.get())
            
            # Évaluer le modèle
            if test_data:
                self.log("\n📊 Évaluation du modèle...")
                scores = tuner.evaluate(test_data)
                self.log(f"Scores: {scores}")
            
            # Sauvegarder le modèle
            tuner.save_model(self.output_dir.get())
            
            # Tester le modèle
            test_texts = [
                "Marie Dupont travaille chez Microsoft.",
                "J'ai rencontré Pierre Martin à Paris.",
                "Le directeur de Google France est basé à Lyon.",
                "Sophie Bernard et Jean Rousseau se retrouvent à l'Université de Paris.",
                "La conférence de Total Energies aura lieu à Marseille."
            ]
            tuner.test_model(test_texts)
            
            self.log("\n✅ Entraînement terminé avec succès!")
            messagebox.showinfo("Succès", "L'entraînement est terminé avec succès!")
            
        except Exception as e:
            self.log(f"\n❌ Erreur: {str(e)}")
            messagebox.showerror("Erreur", f"Une erreur s'est produite:\n{str(e)}")
        finally:
            # Restaurer stdout
            sys.stdout = old_stdout
            self.is_training = False
            self.train_button.config(state='normal')
            self.progress.stop()
    
    def test_model(self):
        """Teste un modèle existant"""
        # Demander quel modèle tester
        test_choice = messagebox.askyesno(
            "Tester un modèle",
            "Voulez-vous tester le modèle que vous venez d'entraîner?\n\n"
            "Oui = Modèle actuel\n"
            "Non = Choisir un autre modèle"
        )
        
        model_path = self.output_dir.get()
        
        if not test_choice:
            # Demander le chemin du modèle
            model_path = filedialog.askdirectory(
                title="Sélectionner le dossier du modèle à tester"
            )
            if not model_path:
                return
        
        # Vérifier que le modèle existe
        if not Path(model_path).exists():
            messagebox.showerror("Erreur", f"Le dossier '{model_path}' n'existe pas!")
            return
            
        if not (Path(model_path) / "meta.json").exists():
            messagebox.showerror("Erreur", 
                               f"Aucun modèle valide trouvé dans '{model_path}'!\n"
                               "Assurez-vous que le dossier contient un modèle spaCy.")
            return
            
        try:
            # Charger le modèle
            self.log(f"Chargement du modèle depuis: {model_path}")
            nlp = spacy.load(model_path)
            
            # Créer la fenêtre de test
            test_window = tk.Toplevel(self.root)
            test_window.title("Tester le modèle Multi-Entités")
            test_window.geometry("700x600")
            
            # Informations sur le modèle
            info_frame = ttk.LabelFrame(test_window, text="Informations du modèle", padding="5")
            info_frame.pack(fill=tk.X, padx=10, pady=5)
            
            model_info = f"Modèle: {nlp.meta.get('name', 'N/A')}\n"
            model_info += f"Version: {nlp.meta.get('version', 'N/A')}\n"
            model_info += f"Langue: {nlp.meta.get('lang', 'N/A')}\n"
            model_info += f"Pipeline: {', '.join(nlp.pipe_names)}\n"
            
            # Récupérer les labels NER
            if 'ner' in nlp.pipe_names:
                ner_labels = nlp.get_pipe('ner').labels
                model_info += f"Entités reconnues: {', '.join(ner_labels)}"
            
            ttk.Label(info_frame, text=model_info, font=('Courier', 10)).pack()
            
            # Zone de saisie
            ttk.Label(test_window, text="Entrez un texte à analyser:", 
                     font=('Helvetica', 12)).pack(pady=10)
            
            text_input = scrolledtext.ScrolledText(test_window, height=5, width=70)
            text_input.pack(padx=10, pady=5)
            text_input.insert(tk.END, "Marie Dupont travaille chez Microsoft à Paris.")
            
            # Zone de résultats
            result_frame = ttk.LabelFrame(test_window, text="Résultats", padding="5")
            result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            result_text = scrolledtext.ScrolledText(result_frame, height=12, width=70)
            result_text.pack(fill=tk.BOTH, expand=True)
            
            def analyze():
                text = text_input.get("1.0", tk.END).strip()
                doc = nlp(text)
                
                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, "Entités détectées:\n" + "="*60 + "\n\n")
                
                if doc.ents:
                    # Regrouper par type d'entité
                    entities_by_type = {}
                    for ent in doc.ents:
                        if ent.label_ not in entities_by_type:
                            entities_by_type[ent.label_] = []
                        entities_by_type[ent.label_].append(ent)
                    
                    # Afficher par type
                    entity_icons = {'PER': '👤', 'ORG': '🏢', 'LOC': '📍'}
                    entity_names = {'PER': 'Personnes', 'ORG': 'Organisations', 'LOC': 'Lieux'}
                    
                    for entity_type, entities in entities_by_type.items():
                        icon = entity_icons.get(entity_type, '📌')
                        name = entity_names.get(entity_type, entity_type)
                        result_text.insert(tk.END, f"\n{icon} {name} ({entity_type}):\n")
                        result_text.insert(tk.END, "-" * 40 + "\n")
                        
                        for ent in entities:
                            result_text.insert(tk.END, f"  • {ent.text}\n")
                            result_text.insert(tk.END, f"    Position: caractères {ent.start_char}-{ent.end_char}\n")
                            
                            # Contexte
                            context_start = max(0, ent.start_char - 20)
                            context_end = min(len(text), ent.end_char + 20)
                            context = text[context_start:context_end]
                            if context_start > 0:
                                context = "..." + context
                            if context_end < len(text):
                                context = context + "..."
                            result_text.insert(tk.END, f"    Contexte: {context}\n\n")
                else:
                    result_text.insert(tk.END, "❌ Aucune entité détectée dans ce texte.")
                
                # Ajouter le texte annoté avec couleurs
                result_text.insert(tk.END, "\n\nTexte annoté:\n" + "="*60 + "\n")
                
                # Créer une copie du texte pour l'annotation
                annotated_parts = []
                last_end = 0
                
                for ent in sorted(doc.ents, key=lambda x: x.start_char):
                    # Ajouter le texte avant l'entité
                    if ent.start_char > last_end:
                        annotated_parts.append(text[last_end:ent.start_char])
                    
                    # Ajouter l'entité annotée
                    entity_colors = {'PER': '🔵', 'ORG': '🟢', 'LOC': '🔴'}
                    color = entity_colors.get(ent.label_, '⚫')
                    annotated_parts.append(f"{color}[{ent.text}|{ent.label_}]")
                    
                    last_end = ent.end_char
                
                # Ajouter le texte restant
                if last_end < len(text):
                    annotated_parts.append(text[last_end:])
                
                result_text.insert(tk.END, ''.join(annotated_parts))
            
            # Boutons
            button_frame = ttk.Frame(test_window)
            button_frame.pack(pady=10)
            
            ttk.Button(button_frame, text="Analyser", command=analyze, 
                      style='Accent.TButton').pack(side=tk.LEFT, padx=5)
            
            # Exemples prédéfinis
            def load_example(text):
                text_input.delete("1.0", tk.END)
                text_input.insert("1.0", text)
                analyze()
            
            examples = [
                "Sophie Bernard de Microsoft France a rencontré Michel Girard à Lyon.",
                "Le rapport de François Lefebvre doit être envoyé à la Banque Nationale de Paris.",
                "Jean-Pierre Martin, directeur de Total Energies, participera à la conférence de Marseille.",
                "L'Université de Paris et le CNRS ont signé un partenariat à Strasbourg.",
                "Catherine Rousseau quittera Google France pour rejoindre Air France à Toulouse."
            ]
            
            example_menu = tk.Menu(test_window, tearoff=0)
            for ex in examples:
                example_menu.add_command(label=ex[:50] + "...", 
                                       command=lambda t=ex: load_example(t))
            
            example_button = ttk.Button(button_frame, text="Exemples ▼")
            example_button.pack(side=tk.LEFT, padx=5)
            example_button.bind("<Button-1>", lambda e: example_menu.post(e.x_root, e.y_root))
            
            # Statistiques
            stats_button = ttk.Button(button_frame, text="📊 Statistiques", 
                                    command=lambda: self.show_model_stats(nlp))
            stats_button.pack(side=tk.LEFT, padx=5)
            
            # Analyser automatiquement le texte par défaut
            analyze()
            
        except Exception as e:
            self.log(f"❌ Erreur lors du chargement du modèle: {str(e)}")
            messagebox.showerror("Erreur", f"Impossible de charger le modèle:\n{str(e)}")
    
    def show_model_stats(self, nlp):
        """Affiche des statistiques sur le modèle"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Statistiques du modèle")
        stats_window.geometry("400x300")
        
        stats_text = scrolledtext.ScrolledText(stats_window, height=15, width=50)
        stats_text.pack(padx=10, pady=10)
        
        stats = "STATISTIQUES DU MODÈLE\n" + "="*40 + "\n\n"
        
        # Métadonnées
        stats += "Métadonnées:\n"
        stats += f"  • Nom: {nlp.meta.get('name', 'N/A')}\n"
        stats += f"  • Version: {nlp.meta.get('version', 'N/A')}\n"
        stats += f"  • Langue: {nlp.meta.get('lang', 'N/A')}\n"
        stats += f"  • Description: {nlp.meta.get('description', 'N/A')}\n\n"
        
        # Pipeline
        stats += "Pipeline:\n"
        for pipe_name in nlp.pipe_names:
            stats += f"  • {pipe_name}\n"
        
        # Labels NER
        if 'ner' in nlp.pipe_names:
            ner = nlp.get_pipe('ner')
            stats += f"\nEntités NER ({len(ner.labels)} types):\n"
            for label in sorted(ner.labels):
                entity_desc = {
                    'PER': 'Personnes',
                    'ORG': 'Organisations', 
                    'LOC': 'Lieux'
                }.get(label, label)
                stats += f"  • {label}: {entity_desc}\n"
        
        stats_text.insert("1.0", stats)
        stats_text.config(state='disabled')
    
    def run(self):
        """Lance l'interface graphique"""
        self.root.mainloop()


class TextRedirector:
    """Redirige la sortie texte vers le widget de log"""
    def __init__(self, text_widget_callback):
        self.callback = text_widget_callback
        
    def write(self, string):
        self.callback(string.rstrip())
        
    def flush(self):
        pass


def main():
    """Point d'entrée principal"""
    # Si des arguments sont passés, utiliser le mode ligne de commande
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Fine-tuning de modèles spaCy pour la NER multi-entités")
        parser.add_argument("--model-size", choices=["sm", "md", "lg"], default="sm",
                           help="Taille du modèle spaCy (sm, md, lg)")
        parser.add_argument("--lang", default="fr", help="Langue du modèle")
        parser.add_argument("--names-file", help="Fichier contenant les noms de personnes")
        parser.add_argument("--orgs-file", help="Fichier contenant les noms d'organisations")
        parser.add_argument("--places-file", help="Fichier contenant les noms de lieux")
        parser.add_argument("--templates-file", required=True, help="Fichier contenant les phrases types")
        parser.add_argument("--output-dir", default="model_multi_ner", help="Répertoire de sortie")
        parser.add_argument("--iterations", type=int, default=30, help="Nombre d'itérations")
        parser.add_argument("--samples-per-template", type=int, default=5, 
                           help="Nombre d'échantillons par phrase type")
        
        args = parser.parse_args()
        
        # Vérifier qu'au moins un fichier d'entités est fourni
        if not any([args.names_file, args.orgs_file, args.places_file]):
            print("❌ Erreur: Au moins un fichier d'entités doit être fourni!")
            sys.exit(1)
        
        # Code en ligne de commande
        tuner = SpacyMultiEntityFineTuner(model_size=args.model_size, lang=args.lang)
        tuner.load_base_model()
        
        # Charger les entités
        if args.names_file:
            tuner.load_entities_from_file(args.names_file, 'PER')
        if args.orgs_file:
            tuner.load_entities_from_file(args.orgs_file, 'ORG')
        if args.places_file:
            tuner.load_entities_from_file(args.places_file, 'LOC')
        
        # Charger les templates
        templates = tuner.load_sentence_templates(args.templates_file)
        
        # Générer les données
        training_data = tuner.generate_training_data(templates, args.samples_per_template)
        tuner.training_data = training_data
        
        # Diviser les données
        split_point = int(len(training_data) * 0.8)
        train_data = training_data[:split_point]
        test_data = training_data[split_point:]
        
        # Entraîner
        training_examples = tuner.prepare_training_data(train_data)
        tuner.train(training_examples, n_iter=args.iterations)
        
        # Évaluer
        if test_data:
            print("\n📊 Évaluation du modèle...")
            scores = tuner.evaluate(test_data)
            print(f"Score NER: {scores}")
        
        # Sauvegarder
        tuner.save_model(args.output_dir)
        
        # Tester
        test_texts = [
            "Marie Dupont travaille chez Microsoft à Paris.",
            "Le rapport de Google France a été présenté à Lyon.",
            "Jean Martin dirige l'Université de Toulouse."
        ]
        tuner.test_model(test_texts)
        
        print("\n✅ Processus terminé avec succès!")
        print(f"\nPour utiliser le modèle:")
        print(f"nlp = spacy.load('{args.output_dir}')")
    else:
        # Lancer l'interface graphique
        app = SpacyMultiEntityFineTunerGUI()
        app.run()


if __name__ == "__main__":
    main()