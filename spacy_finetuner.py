import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import json
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
from io import StringIO


class SpacyFineTuner:
    """Classe pour faciliter le fine-tuning de modèles spaCy pour la NER"""
    
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
    
    def load_proper_names(self, filepath: str) -> List[str]:
        """
        Charge les noms propres depuis un fichier
        
        Args:
            filepath: Chemin vers le fichier de noms propres (un nom par ligne)
        
        Returns:
            Liste des noms propres
        """
        names = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)
        
        print(f"✓ {len(names)} noms propres chargés")
        return names
    
    def load_sentence_templates(self, filepath: str) -> List[str]:
        """
        Charge les phrases types depuis un fichier
        
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
    
    def generate_training_data(self, names: List[str], templates: List[str], 
                             samples_per_template: int = 5) -> List[Tuple[str, Dict]]:
        """
        Génère des données d'entraînement en combinant noms et phrases types
        
        Args:
            names: Liste des noms propres
            templates: Liste des phrases types avec placeholders
            samples_per_template: Nombre d'échantillons à générer par template
        
        Returns:
            Liste de tuples (texte, annotations)
        """
        training_examples = []
        
        for template in templates:
            # Compter le nombre de placeholders dans le template
            placeholders = re.findall(r'\{NOM\}', template)
            num_placeholders = len(placeholders)
            
            if num_placeholders == 0:
                continue
                
            # Générer plusieurs exemples pour chaque template
            for _ in range(samples_per_template):
                # Sélectionner aléatoirement des noms
                selected_names = random.sample(names, min(num_placeholders, len(names)))
                
                # Remplacer les placeholders et créer les annotations
                text = template
                entities = []
                offset = 0
                
                for i, name in enumerate(selected_names):
                    # Trouver la position du placeholder
                    placeholder_pos = text.find('{NOM}', offset)
                    if placeholder_pos == -1:
                        break
                        
                    # Remplacer le placeholder
                    text = text[:placeholder_pos] + name + text[placeholder_pos + 5:]
                    
                    # Ajouter l'annotation
                    start = placeholder_pos
                    end = placeholder_pos + len(name)
                    entities.append((start, end, "PER"))  # PER pour personne
                    
                    # Mettre à jour l'offset pour la prochaine recherche
                    offset = end
                
                # Ajouter l'exemple aux données d'entraînement
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
        
        # Ajouter les labels
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
        
        # Mettre à jour les métadonnées du modèle avant de sauvegarder
        self.nlp.meta["name"] = f"custom_ner_{self.model_size}"
        self.nlp.meta["version"] = "1.0.0"
        self.nlp.meta["description"] = "Modèle spaCy fine-tuné pour la reconnaissance de noms propres"
        self.nlp.meta["author"] = "SpacyFineTuner"
        self.nlp.meta["lang"] = self.lang  # Important pour éviter l'erreur E054
        self.nlp.meta["pipeline"] = list(self.nlp.pipe_names)
        self.nlp.meta["labels"] = {
            "ner": list(self.nlp.get_pipe("ner").labels)
        }
        
        # Sauvegarder le modèle avec ses métadonnées
        self.nlp.to_disk(output_path)
        print(f"\n✓ Modèle sauvegardé dans: {output_path}")
        
        # Créer aussi un fichier config.cfg si nécessaire
        config_path = output_path / "config.cfg"
        if not config_path.exists():
            # Écrire la configuration minimale
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


class SpacyFineTunerGUI:
    """Interface graphique pour le fine-tuning de modèles spaCy"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SpaCy Fine-Tuner - Entraînement NER")
        self.root.geometry("800x600")
        
        # Variables
        self.names_file = tk.StringVar()
        self.templates_file = tk.StringVar()
        self.output_dir = tk.StringVar(value="model_custom_ner")
        self.model_size = tk.StringVar(value="sm")
        self.iterations = tk.IntVar(value=30)
        self.samples_per_template = tk.IntVar(value=5)
        self.is_training = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configuration des poids pour le redimensionnement
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Titre
        title = ttk.Label(main_frame, text="Fine-Tuning SpaCy pour la reconnaissance d'entités", 
                         font=('Helvetica', 16, 'bold'))
        title.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Section fichiers
        files_frame = ttk.LabelFrame(main_frame, text="Fichiers de données", padding="10")
        files_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        files_frame.columnconfigure(1, weight=1)
        
        # Fichier noms propres
        ttk.Label(files_frame, text="Noms propres:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(files_frame, textvariable=self.names_file).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(files_frame, text="Parcourir", 
                  command=lambda: self.browse_file(self.names_file)).grid(row=0, column=2, padx=5)
        
        # Fichier phrases types
        ttk.Label(files_frame, text="Phrases types:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(files_frame, textvariable=self.templates_file).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(files_frame, text="Parcourir", 
                  command=lambda: self.browse_file(self.templates_file)).grid(row=1, column=2, padx=5)
        
        # Section paramètres
        params_frame = ttk.LabelFrame(main_frame, text="Paramètres", padding="10")
        params_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)
        
        # Modèle
        ttk.Label(params_frame, text="Taille du modèle:").grid(row=0, column=0, sticky=tk.W, padx=5)
        model_combo = ttk.Combobox(params_frame, textvariable=self.model_size, 
                                  values=["sm", "md", "lg"], state="readonly", width=10)
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Itérations
        ttk.Label(params_frame, text="Itérations:").grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Spinbox(params_frame, from_=10, to=100, textvariable=self.iterations, 
                   width=10).grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Échantillons par template
        ttk.Label(params_frame, text="Échantillons/template:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(params_frame, from_=1, to=20, textvariable=self.samples_per_template, 
                   width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Répertoire de sortie
        ttk.Label(params_frame, text="Dossier de sortie:").grid(row=1, column=2, sticky=tk.W, padx=5)
        ttk.Entry(params_frame, textvariable=self.output_dir, width=20).grid(row=1, column=3, sticky=(tk.W, tk.E), padx=5)
        
        # Boutons d'action
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.train_button = ttk.Button(action_frame, text="Lancer l'entraînement", 
                                      command=self.start_training, style='Accent.TButton')
        self.train_button.grid(row=0, column=0, padx=5)
        
        ttk.Button(action_frame, text="Tester le modèle", 
                  command=self.test_model).grid(row=0, column=1, padx=5)
        
        # Zone de log
        log_frame = ttk.LabelFrame(main_frame, text="Journal d'exécution", padding="10")
        log_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Barre de progression
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
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
    
    def start_training(self):
        """Lance l'entraînement dans un thread séparé"""
        if self.is_training:
            messagebox.showwarning("Entraînement en cours", 
                                 "Un entraînement est déjà en cours!")
            return
            
        # Vérifier les fichiers
        if not self.names_file.get() or not self.templates_file.get():
            messagebox.showerror("Erreur", 
                               "Veuillez sélectionner les deux fichiers requis!")
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
            self.log("Initialisation du fine-tuner...")
            tuner = SpacyFineTuner(model_size=self.model_size.get(), lang="fr")
            
            # Charger le modèle de base
            self.log("Chargement du modèle de base...")
            tuner.load_base_model()
            
            # Charger les données
            self.log("Chargement des données...")
            names = tuner.load_proper_names(self.names_file.get())
            templates = tuner.load_sentence_templates(self.templates_file.get())
            
            # Générer les données d'entraînement
            self.log("Génération des données d'entraînement...")
            training_data = tuner.generate_training_data(
                names, templates, self.samples_per_template.get()
            )
            tuner.training_data = training_data
            
            # Diviser en ensemble d'entraînement et de test
            split_point = int(len(training_data) * 0.8)
            train_data = training_data[:split_point]
            test_data = training_data[split_point:]
            
            # Préparer les données d'entraînement
            self.log("Préparation des données...")
            training_examples = tuner.prepare_training_data(train_data)
            
            # Entraîner le modèle
            tuner.train(training_examples, n_iter=self.iterations.get())
            
            # Évaluer le modèle
            if test_data:
                self.log("\nÉvaluation du modèle...")
                scores = tuner.evaluate(test_data)
                self.log(f"Scores: {scores}")
            
            # Sauvegarder le modèle
            tuner.save_model(self.output_dir.get())
            
            # Tester le modèle
            test_texts = [
                "Marie Dupont travaille chez Microsoft.",
                "J'ai rencontré Pierre Martin hier au bureau.",
                "Le docteur Jean-Paul Rousseau est en consultation."
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
            test_window.title("Tester le modèle")
            test_window.geometry("600x500")
            
            # Informations sur le modèle
            info_frame = ttk.LabelFrame(test_window, text="Informations du modèle", padding="5")
            info_frame.pack(fill=tk.X, padx=10, pady=5)
            
            model_info = f"Modèle: {nlp.meta.get('name', 'N/A')}\n"
            model_info += f"Version: {nlp.meta.get('version', 'N/A')}\n"
            model_info += f"Langue: {nlp.meta.get('lang', 'N/A')}\n"
            model_info += f"Pipeline: {', '.join(nlp.pipe_names)}"
            
            ttk.Label(info_frame, text=model_info, font=('Courier', 10)).pack()
            
            # Zone de saisie
            ttk.Label(test_window, text="Entrez un texte à analyser:", 
                     font=('Helvetica', 12)).pack(pady=10)
            
            text_input = scrolledtext.ScrolledText(test_window, height=5, width=60)
            text_input.pack(padx=10, pady=5)
            text_input.insert(tk.END, "Marie Dupont a envoyé un email à Pierre Martin.")
            
            # Zone de résultats
            result_frame = ttk.LabelFrame(test_window, text="Résultats", padding="5")
            result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            result_text = scrolledtext.ScrolledText(result_frame, height=10, width=60)
            result_text.pack(fill=tk.BOTH, expand=True)
            
            def analyze():
                text = text_input.get("1.0", tk.END).strip()
                doc = nlp(text)
                
                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, "Entités détectées:\n" + "="*50 + "\n\n")
                
                if doc.ents:
                    for ent in doc.ents:
                        result_text.insert(tk.END, f"📌 {ent.text}\n")
                        result_text.insert(tk.END, f"   Type: {ent.label_}\n")
                        result_text.insert(tk.END, f"   Position: caractères {ent.start_char}-{ent.end_char}\n")
                        result_text.insert(tk.END, f"   Contexte: ...{text[max(0, ent.start_char-20):ent.end_char+20]}...\n\n")
                else:
                    result_text.insert(tk.END, "❌ Aucune entité détectée dans ce texte.")
                
                # Ajouter le texte annoté
                result_text.insert(tk.END, "\nTexte annoté:\n" + "="*50 + "\n")
                annotated_text = text
                offset = 0
                for ent in sorted(doc.ents, key=lambda x: x.start_char):
                    start = ent.start_char + offset
                    end = ent.end_char + offset
                    annotation = f"[{ent.text}|{ent.label_}]"
                    annotated_text = annotated_text[:start] + annotation + annotated_text[end:]
                    offset += len(annotation) - (end - start)
                
                result_text.insert(tk.END, annotated_text)
            
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
                "Sophie Bernard a rencontré Michel Girard hier.",
                "Le rapport de François Lefebvre doit être validé par Catherine Rousseau.",
                "Jean-Pierre Martin et Isabelle Moreau participent à la réunion."
            ]
            
            example_menu = tk.Menu(test_window, tearoff=0)
            for ex in examples:
                example_menu.add_command(label=ex[:50] + "...", 
                                       command=lambda t=ex: load_example(t))
            
            example_button = ttk.Button(button_frame, text="Exemples ▼")
            example_button.pack(side=tk.LEFT, padx=5)
            example_button.bind("<Button-1>", lambda e: example_menu.post(e.x_root, e.y_root))
            
            # Analyser automatiquement le texte par défaut
            analyze()
            
        except Exception as e:
            self.log(f"❌ Erreur lors du chargement du modèle: {str(e)}")
            messagebox.showerror("Erreur", f"Impossible de charger le modèle:\n{str(e)}")
    
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
        parser = argparse.ArgumentParser(description="Fine-tuning de modèles spaCy pour la NER")
        parser.add_argument("--model-size", choices=["sm", "md", "lg"], default="sm",
                           help="Taille du modèle spaCy (sm, md, lg)")
        parser.add_argument("--lang", default="fr", help="Langue du modèle")
        parser.add_argument("--names-file", required=True, help="Fichier contenant les noms propres")
        parser.add_argument("--templates-file", required=True, help="Fichier contenant les phrases types")
        parser.add_argument("--output-dir", default="model_custom_ner", help="Répertoire de sortie")
        parser.add_argument("--iterations", type=int, default=30, help="Nombre d'itérations")
        parser.add_argument("--samples-per-template", type=int, default=5, 
                           help="Nombre d'échantillons par phrase type")
        
        args = parser.parse_args()
        
        # Code original en ligne de commande
        tuner = SpacyFineTuner(model_size=args.model_size, lang=args.lang)
        tuner.load_base_model()
        names = tuner.load_proper_names(args.names_file)
        templates = tuner.load_sentence_templates(args.templates_file)
        training_data = tuner.generate_training_data(names, templates, args.samples_per_template)
        tuner.training_data = training_data
        
        split_point = int(len(training_data) * 0.8)
        train_data = training_data[:split_point]
        test_data = training_data[split_point:]
        
        training_examples = tuner.prepare_training_data(train_data)
        tuner.train(training_examples, n_iter=args.iterations)
        
        if test_data:
            print("\n📊 Évaluation du modèle...")
            scores = tuner.evaluate(test_data)
            print(f"Score NER: {scores}")
        
        tuner.save_model(args.output_dir)
        
        test_texts = [
            "Marie Dupont travaille chez Microsoft.",
            "J'ai rencontré Pierre Martin hier au bureau.",
            "Le docteur Jean-Paul Rousseau est en consultation."
        ]
        tuner.test_model(test_texts)
        
        print("\n✅ Processus terminé avec succès!")
        print(f"\nPour utiliser le modèle dans une autre application:")
        print(f"nlp = spacy.load('{args.output_dir}')")
    else:
        # Lancer l'interface graphique
        app = SpacyFineTunerGUI()
        app.run()


if __name__ == "__main__":
    main()