import dill as pickle
import scraper
from lime.lime_text import LimeTextExplainer


def saved_model(news_test, arguments):

    if arguments.NBmodel:
        filename = 'saved_models/model_nb.pk'

    elif arguments.DTmodel:
        filename = 'saved_models/model_dt.pk'

    elif arguments.SVMmodel:
        filename = 'saved_models/model_svm.pk'

    elif arguments.SNNmodel:
        # filename = 'saved_models/model_sk.pk'
        filename = 'saved_models/model_snn_sk.pk'

    else:
        print('Wrong parameter! Available : --SNN --NB --SV --DT')
        exit(-1)

    try:
        with open(filename, 'rb') as f:
            loaded_pipeline = pickle.load(f)
    except FileNotFoundError:
        print('Model does not exist! Use --s to save new model or no second parameter.')
        exit(-1)

    if arguments.url:
        online_article = scraper.parse_article(arguments.url)

        prediction = loaded_pipeline.predict_proba(online_article)
        prediction_label = loaded_pipeline.predict(online_article)

        # Lime explanation
        explainer = LimeTextExplainer(class_names=['True', 'False'], bow=True)
        explanation = explainer.explain_instance(online_article[0],
                                                 loaded_pipeline.predict_proba,
                                                 num_features=20
                                                 )
        explanation.save_to_file('html/article_lime_explanation.html')

    else:
        prediction = loaded_pipeline.predict_proba(news_test)
        prediction_label = loaded_pipeline.predict(news_test)

    for i in range(0, len(prediction), 2):

        print('\033[91m' + 'Fake: ' + '\033[0m' + "%.2f" % (prediction.item(i)*100.0) + ' % -- ' +
              '\033[92m' + 'Real: ' + '\033[0m' + "%.2f" % (prediction.item(i+1)*100.0) + ' %')
        print('------------------------------')
        print(prediction_label.item(i))

        if prediction_label.item(i) == 1:
            print(" -- True -- ")
        else:
            print(" -- False -- ")
