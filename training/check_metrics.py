from evaluate import summary
rows=[{'logits':[3.,0.],'label':0,'semantic_label':'A','task':'choice_task','qtype':'choice'}, {'logits':[3.,0.],'label':1,'semantic_label':'A','task':'choice_task','qtype':'choice'}, {'logits':[3.,0.],'label':0,'semantic_label':'B','task':'choice_task','qtype':'choice'}]
r=summary(rows,1.);assert r['accuracy']==2/3;assert r['mean_class_recall']==.75,r
print('PASS: balanced recall uses semantic answers, not randomized option positions')
