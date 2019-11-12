from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import sleep
from time import time
from datetime import datetime
import model2_config
import params_printer

def execute():

    model2_config.init_params()
    params_printer.output()

    import train_image_classifier
    import eval_image_classifier_original
    import eval_image_classifier
    import validation_confusion_matrix
    import export_to_saved_model

    #"""
    print("Begin %s" % format(datetime.now().isoformat()))
    t_begin = time()

    print("Begin model2 train ...")
    t1 = time()
    train_image_classifier.train()
    t2 = time()
    print("Model2 end train %d s" % (t2-t1))

    sleep(1)

    print("Begin model2 validations and find a lucky check point ...")
    t1 = time()
    eval_image_classifier_original.print_train_acc()
    max_checkpoint_path = eval_image_classifier.find_max_accuracy_checkpoint()
    validation_confusion_matrix.execute(max_checkpoint_path, "model2")

    t2 = time()
    print("Model2 end validations %d s" %(t2 - t1))

    print("Begin model2 export to saved model ...")
    t1 = time()
    export_to_saved_model.export(max_checkpoint_path, "model2")
    t2 = time()
    print("End model2 export to saved model %d s" %(t2 - t1))

    print("Done of all %s" % format(datetime.now().isoformat()))
    t_end = time()

    print("Model2 all end: %d" % (t_end - t_begin))
    #"""
