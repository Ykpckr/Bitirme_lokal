def evaluate(preds, gts, tol=5):
    tp, fp = 0,0
    for p in preds:
        if any(abs(p[0]-g)<tol for g in gts):
            tp+=1
        else:
            fp+=1
    return tp/(tp+fp+1e-6)
