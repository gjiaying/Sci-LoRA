import pandas as pd
import numpy as np
import json
import logging
import sys
import os
import ast
from collections import Counter
import networkx as nx
from torchmetrics.text.bert import BERTScore
from torchmetrics.text import BLEUScore
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import Levenshtein
from transformers import T5Tokenizer, T5ForConditionalGeneration
#from bart_score import BARTScorer
import torch
from blonde import BLONDE
from evaluate import load
import textstat
import os
import nltk
import spacy
#from modelscope.pipelines import pipeline
#from modelscope.utils.constant import Tasks
#from modelscope.models.nlp.unite.configuration import InputFormat
from lexical_diversity import lex_div as ld
from nltk import ngrams
from nltk.probability import FreqDist
from scipy.spatial.distance import jensenshannon
import syllables
#from sle.scorer import SLEScorer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#ground_truth_path = './data/results/plos_phi35.csv'
#generated_path = './data/results/elife_phi35.json'
#log_file_path = './data/logs/phi35_plos.log'

directory_path = './data/results_lora/'
file_names = os.listdir(directory_path)


def read_file_csv(file):
    df = pd.read_csv(file)
    values = df['Input'].tolist()#source
    sources = df['Labels'].tolist()#ground truth
    outputs = df['Generated_Output'].apply(str).tolist()
    
    value,source,generated_output = [],[],[]
    for item in values:
        value.append(item)
    for item in sources:
        source.append(item)
    for item in outputs:
        generated_output.append(item)
    #return source, value, generated_output
    return value, source, generated_output

def read_file_json_LLaMA(file):
    with open(file, 'r') as file:
        data = json.load(file)
    return data

def read_file_json(file):
    with open(file, 'r') as file:
        data = json.load(file)
    generated_value = []
    for value in data.values():
        generated_value.append(value)
    return generated_value

def paragraphs_to_sentences(ground_truth, generated_value):
    nltk.download('punkt')
    ground, generated = [],[]
    for i in range(len(ground_truth)):
        new_ground = nltk.sent_tokenize(ground_truth[i])
        new_generated = nltk.sent_tokenize(generated_value[i])
        if len(new_ground) >= len(new_generated):
            new_ground = new_ground[:len(new_generated)]
        else:
            new_generated = new_generated[:len(new_ground)]    
        ground.extend(new_ground)
        generated.extend(new_generated)
    return ground, generated
 
def bertScore(ground_truth, generated_value):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=generated_value, references=ground_truth, model_type="distilbert-base-uncased")
    return results['f1'], results['precision'], results['recall']

def bartScore(ground_truth, generated_value):
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    result = bart_scorer.score(ground_truth, generated_value, batch_size=4)
    print(result)
    
    
def d_BLEU(ground_truth, generated_value):
    bleu = load("bleu")
    results = bleu.compute(predictions=generated_value, references=ground_truth)
    return results['bleu']
    
def smooth_BLEU(ground_truth, generated_value):
    bleu = load("bleu")
    results = bleu.compute(predictions=generated_value, references=ground_truth, smooth=True)
    return results['bleu']

def s_BLEU(ground_truth, generated_value):
    bleu = load("bleu")
    ground_truth, generated_value = paragraphs_to_sentences(ground_truth, generated_value)
    results = bleu.compute(predictions=generated_value, references=ground_truth)
    return results['bleu']

def comet(ground_truth, generated_value, source):
    comet_model = load('comet')
    results = comet_model.compute(predictions=generated_value, references=ground_truth, sources=source)
    return results['mean_score']

def rouge(ground_truth, generated_value):
    rouge_metric = load('rouge')
    results = rouge_metric.compute(predictions=generated_value, references=ground_truth)
    return results['rouge1'], results['rouge2'], results['rougeL'], results['rougeLsum']

def chrf(ground_truth, generated_value):
    chrf_metric = load('chrf')
    results = chrf_metric.compute(predictions=generated_value, references=ground_truth)
    return results['score']

def inconsistency(ground_truth, generated_value):
    chrf_metric = load('chrf')
    count = 0
    ground_truth, generated_value = paragraphs_to_sentences(ground_truth, generated_value)
    for i in range(len(ground_truth)):
        if len(ground_truth[i]) >= len(generated_value[i]):
            ground_truth[i] = ground_truth[i][:len(generated_value[i])]
        else:
            generated_value[i] = generated_value[i][:len(ground_truth[i])]
        results = chrf_metric.compute(predictions=generated_value[i], references=ground_truth[i])
        if results['score'] < 75:
            count = count + 1
    return count / len(ground_truth)


def meteor(ground_truth, generated_value):
    meteor_metric = load('meteor')
    results = meteor_metric.compute(predictions=generated_value, references=ground_truth)
    return results['meteor']

def ter(ground_truth, generated_value):
    ter_metric = load('ter')
    results = ter_metric.compute(predictions=generated_value, references=ground_truth, case_sensitive=False)
    return results['score']

def sacrebleu(ground_truth, generated_value):
    sacrebleu_metric = load('sacrebleu')
    results = sacrebleu_metric.compute(predictions=generated_value, references=ground_truth)
    return results['score']

def GoogleBLEU(ground_truth, generated_value):
    google_bleu = load("google_bleu")
    results = google_bleu.compute(predictions=generated_value, references=ground_truth)
    return results['google_bleu']

def bleurt(ground_truth, generated_value):
    bleurt = load("bleurt", module_type="metric")
    results = bleurt.compute(predictions=generated_value, references=ground_truth)
    return sum(results['scores']) / len(results['scores'])

def overlap(ground_truth, generated_value):
    words_list1 = [word for sentence in ground_truth for word in sentence.split()]
    words_list2 = [word for sentence in generated_value for word in sentence.split()]
    union_words = set(words_list1).union(set(words_list2))
    overlap_words = set(words_list1).intersection(set(words_list2))
    return 1 - len(overlap_words)/len(union_words)

def deviation_diversity(ground_truth, generated_value):
    results = []
    for i in range(len(generated_value)):
        for j in range(i+1, len(generated_value)):
            if len(generated_value[i]) >= len(generated_value[j]):
                value_i = generated_value[i][:len(generated_value[j])]
                value_j = generated_value[j]
            else:
                value_j = generated_value[j][:len(generated_value[i])]
                value_i = generated_value[i]
            results.append(1-s_BLEU(value_i, value_j))
    return sum(results)/(len(generated_value)*(len(generated_value)-1)/2)

def WER(ground_truth, generated_value):
    wer_metric = load("wer")
    results = []
    for i in range(len(ground_truth)):
        result = wer_metric.compute(references=[ground_truth[i]], predictions=[generated_value[i]])
        results.append(result)
    return sum(results) / len(results)

def CER(ground_truth, generated_value):
    cer_metric = load("cer")
    cer_score = cer_metric.compute(predictions=generated_value, references=ground_truth)
    return cer_score


def frugalScore(ground_truth, generated_value):
    frugalscore = load("frugalscore")
    results = frugalscore.compute(predictions=generated_value, references=ground_truth)
    return sum(results['scores']) / len(results['scores'])

def mauve(ground_truth, generated_value):
    mauve_metric = load('mauve')
    mauve_results = mauve_metric.compute(predictions=generated_value, references=ground_truth)
    return mauve_results.mauve

def Jaccard_Similarity(ground_truth, generated_value):
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))
    ground_truth = ', '.join(ground_truth)
    generated_value = ', '.join(generated_value)
    tokens1 = set(word_tokenize(ground_truth.lower())) - stop_words
    tokens2 = set(word_tokenize(generated_value.lower())) - stop_words
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    jaccard_similarity = intersection / union
    return jaccard_similarity

def Levenshtein_Distance(ground_truth, generated_value):
    sentence1 = ', '.join(ground_truth)
    sentence2 = ', '.join(generated_value)
    lev_distance = Levenshtein.distance(sentence1, sentence2)
    return lev_distance

def Leveshtein_Similarity(ground_truth, generated_value):
    sentence1 = ', '.join(ground_truth)
    sentence2 = ', '.join(generated_value)
    result = Levenshtein.ratio(sentence1, sentence2)
    return result

def cosine_(ground_truth, generated_value):
    sentence1 = ', '.join(ground_truth)
    sentence2 = ', '.join(generated_value)
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([sentence1, sentence2])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity

def A_Score(ground_truth, generated_value):
    result = []
    ground_truth, generated_value = paragraphs_to_sentences(ground_truth, generated_value)
    for i in range(len(ground_truth)):
        words1 = nltk.word_tokenize(ground_truth[i])
        words2 = nltk.word_tokenize(generated_value[i])
        common_words = set(words1) & set(words2)
        result.append(len(common_words) / (len(set(words1)) + len(set(words2))))
    return sum(result) / len(result)
        

def Perplexity(source, ground_truth, model_name='T5'):
    if model_name == 'T5':
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
    else:
        print('Please provide model name.')
        return None
    ppl = []
    for i in range(len(ground_truth)):
        input_ids = tokenizer(source[i], return_tensors="pt").input_ids
        labels = tokenizer(ground_truth[i], return_tensors="pt").input_ids
        loss = model(input_ids=input_ids, labels=labels).loss
        ppl.append(torch.exp(loss))
    return sum(ppl) / len(ppl)

def Perplexity_hugging(source, model):
    input = []
    for item in source:
        input.extend(item[0:1024])
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(model_id=model,
                             add_start_token=True,
                             predictions=input)
    return results["mean_perplexity"]

def TED(ground_truth, generated_value):
    ground_truth, generated_value = paragraphs_to_sentences(ground_truth, generated_value)
    jar_path = './stanford-corenlp-4.2.2/stanford-corenlp-4.2.2.jar'
    models_jar_path = './tools/stanford-corenlp-4.2.2-models-english.jar'
    parser = StanfordDependencyParser(path_to_jar = jar_path, path_to_models_jar = models_jar_path)
    tree_edit_distance = []
    for i in range(len(ground_truth)):
        try:
            result1 = parser.raw_parse(ground_truth[i])
            result2 = parser.raw_parse(generated_value[i])
            dependency1 = result1.__next__()
            dependency2 = result2.__next__()
            G1 = dependency1.nx_graph().reverse()
            G2 = dependency2.nx_graph().reverse()
            tree_edit_distance.append(nx.graph_edit_distance(G1, G2)) 
        except:
            continue
    return sum(tree_edit_distance) / len(tree_edit_distance)

def SARI(ground_truth, generated_value,source):
    wiki_split = load("wiki_split")
    results = wiki_split.compute(sources=source, predictions=generated_value, references=ground_truth)
    return results['sari']

def BLONDE_Score(ground_truth, generated_value):
    blonde = BLONDE()
    score = blonde.corpus_score([generated_value], [[ground_truth]])
    return score

def Unite(ground_truth, generated_value, source):
    input = {
        'hyp': generated_value,
        'src': source,
        'ref': ground_truth
    }
    pipeline_ins = pipeline(task=Tasks.translation_evaluation, model='damo/nlp_unite_up_translation_evaluation_English_large')
    return sum(pipeline_ins(input)['score']) / len(pipeline_ins(input)['score'])

def MLTD(generated_value):
    generated_value = ', '.join(generated_value)
    flt = ld.flemmatize(generated_value)
    return ld.mtld(flt)


def get_ngram_frequencies(text, n):
    words = word_tokenize(text)
    ngrams_list = list(ngrams(words, n))
    freq_dist = FreqDist(ngrams_list)
    return freq_dist

def js_divergence(text1, text2, n):
    freq_dist1 = get_ngram_frequencies(text1, n)
    freq_dist2 = get_ngram_frequencies(text2, n)
    all_ngrams = list(set(list(freq_dist1.keys()) + list(freq_dist2.keys())))
    prob_dist1 = np.array([freq_dist1[ngram] / sum(freq_dist1.values()) for ngram in all_ngrams])
    prob_dist2 = np.array([freq_dist2[ngram] / sum(freq_dist2.values()) for ngram in all_ngrams])
    js_div = jensenshannon(prob_dist1, prob_dist2)
    return js_div
    
def JS_Divergence(ground_truth, generated_value):
    n_value = 2
    result = []
    for i in range(len(ground_truth)):
        if len(ground_truth[i]) >= len(generated_value[i]):
            ground_truth[i] = ground_truth[i][:len(generated_value[i])]
        else:
            generated_value[i] = generated_value[i][:len(ground_truth[i])]
        js_divergence_value = js_divergence(ground_truth[i], generated_value[i], n_value)
        result.append(js_divergence_value)   
    return sum(result) / len(result)

def Halluciation(source, generated_value):
    source_text = ', '.join(source)
    generated_text = ', '.join(generated_value)
    source_tokens = set(word_tokenize(source_text.lower()))
    generated_tokens = set(word_tokenize(generated_text.lower()))
    hallucinated_tokens = generated_tokens - source_tokens
    hallucination_rate = len(hallucinated_tokens) / len(generated_tokens)
    return hallucination_rate

def Toxicity(generated_value):
    toxicity = load("toxicity", module_type="measurement")
    results = toxicity.compute(predictions=generated_value)
    result_2 = toxicity.compute(predictions=generated_value, aggregation="maximum")
    return sum(results['toxicity']) / len(results['toxicity']), result_2['max_toxicity']

def Readability(generated_value):
    text = ', '.join(generated_value)
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum([syllables.estimate(word) for word in words])  # Use syllables.estimate
    # Flesch Reading Ease Formula
    reading_ease = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    return reading_ease

def count_sentence_splits(ground_truth, generated_value):
    result = []
    for i in range(len(ground_truth)):
        result.append(len(generated_value[i])/len(ground_truth[i]))
    return sum(result) / len(result)

def Compression_Ratio(ground_truth, generated_value):
    result = []
    ground_truth, generated_value = paragraphs_to_sentences(ground_truth, generated_value)
    for i in range(len(ground_truth)):
        result.append(len(generated_value[i])/len(ground_truth[i]))
    return sum(result) / len(result)

def Exact_Match(ground_truth, generated_value):
    count = 0
    ground_truth, generated_value = paragraphs_to_sentences(ground_truth, generated_value)
    for i in range(len(ground_truth)):
        if generated_value[i] == ground_truth[i]:
            count = count + 1
    return count / len(ground_truth)

def get_additions_proportion(ground_truth, generated_value):
    ground_truth = ', '.join(ground_truth)
    generated_text = ', '.join(generated_value)
    n_additions = sum((Counter(generated_text.split()) - Counter(ground_truth.split())).values())
    ground_truth = word_tokenize(ground_truth)
    generated_text = word_tokenize(generated_text)
    return n_additions / max(len(ground_truth), len(generated_text))

def get_deletions_proportion(ground_truth, generated_value):
    ground_truth = ', '.join(ground_truth)
    generated_text = ', '.join(generated_value)
    n_additions = sum((Counter(ground_truth.split()) - Counter(generated_text.split())).values())
    ground_truth = word_tokenize(ground_truth)
    generated_text = word_tokenize(generated_text)
    return n_additions / max(len(ground_truth), len(generated_text))

def Greedy_Matching(ground_truth, generated_value):
    nlp = spacy.load("en_core_web_md")
    ground_truth = ', '.join(ground_truth)
    generated_value = ', '.join(generated_value)
    tokens1 = list(nlp(ground_truth))
    tokens2 = list(nlp(generated_value))
    total_similarity_score = 0.0
    for token1 in tokens1:
        max_similarity = 0.0
        for token2 in tokens2:
            embedding1 = token1.vector.reshape(1, -1)
            embedding2 = token2.vector.reshape(1, -1)
            similarity_score = cosine_similarity(embedding1, embedding2)[0, 0]
            max_similarity = max(max_similarity, similarity_score)
        total_similarity_score += max_similarity
    average_similarity_score = total_similarity_score / len(tokens1)
    return average_similarity_score

def Embedding_Average(ground_truth, generated_value):
    nlp = spacy.load("en_core_web_md")
    ground_truth = ', '.join(ground_truth)
    generated_value = ', '.join(generated_value)
    embeddings1 = [token.vector for token in nlp(ground_truth)]
    embeddings2 = [token.vector for token in nlp(generated_value)]
    embedding_average1 = np.mean(embeddings1, axis=0).reshape(1, -1)
    embedding_average2 = np.mean(embeddings2, axis=0).reshape(1, -1)
    return cosine_similarity(embedding_average1, embedding_average2)[0, 0]

def Vector_Extrema(ground_truth, generated_value):
    nlp = spacy.load("en_core_web_md")
    ground_truth = ', '.join(ground_truth)
    generated_value = ', '.join(generated_value)
    embeddings1 = [token.vector for token in nlp(ground_truth)]
    embeddings2 = [token.vector for token in nlp(generated_value)]
    embeddings1 = np.max(embeddings1, axis=0).reshape(1, -1)
    embeddings2 = np.max(embeddings2, axis=0).reshape(1, -1)
    return cosine_similarity(embeddings1, embeddings2)[0, 0]

def Readability_others(generated_value):
    generated_value = ', '.join(generated_value)
    return textstat.flesch_reading_ease(generated_value), textstat.flesch_kincaid_grade(generated_value), textstat.gunning_fog(generated_value), textstat.smog_index(generated_value), textstat.automated_readability_index(generated_value), textstat.coleman_liau_index(generated_value), textstat.linsear_write_formula(generated_value), textstat.dale_chall_readability_score(generated_value), textstat.text_standard(generated_value, float_output=False)

def SLE(generated_value):
    scorer = SLEScorer("liamcripwell/sle-base")
    results = scorer.score(generated_value)
    return sum(results['sle']) / len(results['sle'])



def main():
    for file in file_names:
        log_file_path = f'./data/log_multi/{file[:-4]}.log'
        ground_truth_path = f'./data/results_lora/{file}'
        sys.stdout = open(log_file_path, 'a+')

        source, ground_truth, generated_value = read_file_csv(ground_truth_path) #list form, each abstract is generated_value[i]

        #s_BLEU
        s_bleu = s_BLEU(ground_truth, generated_value)
        print("Sentence-level BLEU:", s_bleu)
        #print('Deviation:', 1-s_bleu)

        d_Bleu = d_BLEU(ground_truth, generated_value)
        #d_Bleu_S = d_BLEU(source, generated_value)
        print('document-level BLEU:', d_Bleu)

        #BertScore
        bertscore_f1, bertscore_precision, bertscore_recall = bertScore(ground_truth, generated_value)
        print('BertScore F1:', sum(bertscore_f1)/len(bertscore_f1))
        #print('BertScore Precision:', sum(bertscore_precision)/len(bertscore_precision))
        #print('BertScore Recall:', sum(bertscore_recall)/len(bertscore_recall))

        #BLONDE
        blonde = BLONDE_Score(ground_truth, generated_value)
        print('BLONDE:', blonde)

        #ROUGE
        rouge1, rouge2, rougeL, rougeLsum = rouge(ground_truth, generated_value)
        print('ROUGE1:', rouge1)
        print('ROUGE2:', rouge2)
        #print('ROUGEL:', rougeL)
        #print('ROUGELSUM:', rougeLsum)

        #meteor
        meteor_score = meteor(ground_truth, generated_value)
        print('Meteor:', meteor_score)


        #Comet
        comet_value = comet(ground_truth, generated_value, source)
        print('COMET value:', comet_value)


        #SARI
        wiki_score = SARI(ground_truth, generated_value, source)
        print('SARI:', wiki_score)

        
        
        #1-∩/∪(↑) OR LTCR
        #overlap_score = overlap(ground_truth, generated_value)
        #print('1-∩/∪(↑):', overlap_score)
        
        #BartScore
        #bartscore = bartScore(ground_truth, generated_value)

    
        #smooth_bleu
        #smooth_bleu = smooth_BLEU(ground_truth, generated_value)
        #print('Smooth BLEU:', smooth_bleu)

        

        #Google BLEU
        #google_bleu = GoogleBLEU(ground_truth, generated_value)
        #print('Google BLEU:', google_bleu)

        #BLEURT #need to run on Colab
        #bleurt_score = bleurt(ground_truth, generated_value)
        #print('Bluert:', bleurt_score)

        #lexiccal diversity:
        #1-BLEU:
        #print('1-BLEU:', 1-d_Bleu)
        

        #chrf
        #chrf_score = chrf(ground_truth, generated_value)
        #print('Chrf:', chrf_score)

        #Inconsistency
        #inconsistency_score = inconsistency(ground_truth, generated_value)
        #print('Inconsistency:', inconsistency_score)

        #CER
        #cer_score = CER(ground_truth, generated_value)
        #print('CER:', cer_score)


        #Jaccard Similarity
        #jaccard_score = Jaccard_Similarity(ground_truth, generated_value)
        #print('Jaccard Similarity:', jaccard_score)

        #Levenshtein Distance
        #levenshtein_score = Levenshtein_Distance(ground_truth, generated_value)
        #print('Levenshtein Score:', levenshtein_score)

        #Leveshtein Similarity
        #leveshtein_similarity = Leveshtein_Similarity(ground_truth, generated_value)
        #print('Levenshtein Similarity:', leveshtein_similarity)

        #Cosine Similarity
        #cosine_score = cosine_(ground_truth, generated_value)
        #print('Cosine Similarity:', cosine_score)

        #WER
        #wer = WER(ground_truth, generated_value)
        #print('WER:', wer)

        #TED-F needs long time for computation
        #ted_f = TED(ground_truth, generated_value)
        #print('TED-F:', ted_f)

        #Perplexity Need to calculate during training
        #perplexity = Perplexity(source, ground_truth, model_name)
        #print('Perplexity:', perplexity.item())

        #Perplexity - Huggingface
        #perplexity_score = Perplexity_hugging(source, 'gpt2')
        #print('Perplexity Score from Huggingface:', perplexity_score)

        #A-Score
        #a_score = A_Score(ground_truth, generated_value)
        #print('A Score:', a_score)


        #Unite run on colab
        #unite_score = Unite(ground_truth, generated_value, source)
        #print('Unite Score:', unite_score)

        #MAUVE
        #mauve_score = mauve(ground_truth, generated_value)
        #print('MAUVE:', mauve_score)

        #MLTD
        #mltd_value = MLTD(generated_value)
        #print('MLTD:', mltd_value)

        #JS Divergence
        #js_divergence = JS_Divergence(ground_truth, generated_value)
        #print('JS:', js_divergence)
        
        #Halluciation Rate
        #halluciation_rate = Halluciation(source, generated_value)
        #print('Halluciation Rate:', halluciation_rate)

        #Readability others
        #scores, grade, fog, smog, ARI, cole, lwf, dale, con = Readability_others(generated_value)
        #print('Flesch Reading Ease Score:', scores)
        #print('Flesch-Kincaid Grade:', grade)
        #print('The Fog Scale:', fog)
        #print('SMOG:', smog)
        #print('ARI:', ARI)
        #print('The Coleman-Liau Index:', cole)
        #print('Linsear Write Formula:', lwf)
        #print('Dale-Chall Readability Score:', dale)
        #print('Readability Consensus:', con)

        #Toxicity
        toxicity_mean, toxicity_max = Toxicity(generated_value)
        print('Mean Toxicity:', toxicity_mean)
        #print('Max Toxicity:', toxicity_max)

        #sacrebleu
        #sacrebleu_score = sacrebleu(ground_truth, generated_value)
        #print('SacreBLEU:', sacrebleu_score)

        #frugalScore
        #frugalscore = frugalScore(ground_truth, generated_value)
        #print('frugalScore:', frugalscore)

        #Readability flesch_reading_ease
        readability = Readability(generated_value)
        print('Readability:', readability)
        
        #count_sentence_splits
        #sentence_splits = count_sentence_splits(ground_truth, generated_value)
        #print('Count Sentence Splits:', sentence_splits)

        #Compression Ratio
        #compression_ratio = Compression_Ratio(ground_truth, generated_value)
        #print('Compression Ratio:', compression_ratio)

        #Exact Match Ratio - Sentence LEvel
        #exact_value = Exact_Match(ground_truth, generated_value)
        #print('Exact Match:', exact_value)

        #get_additions_proportion
        #addition_proportion = get_additions_proportion(ground_truth, generated_value)
        #print('Addition Proportion:', addition_proportion)

        #get_deletions_proportion
        #deletion_proportion = get_deletions_proportion(ground_truth, generated_value)
        #print('Deletion Proportion:', deletion_proportion)

        #Vector Extrema
        #vector_extrema = Vector_Extrema(ground_truth, generated_value)
        #print('Vector Extrema:', vector_extrema)

        #Embedding Average
        #embedding_average = Embedding_Average(ground_truth, generated_value)
        #print('Embedding Average:', embedding_average)
    
    
if __name__ == "__main__":
    main()