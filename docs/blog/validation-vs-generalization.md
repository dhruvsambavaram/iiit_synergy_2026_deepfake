# Validation ≠ Generalization

_The surprisingly easy ways to fool yourself into thinking your model has learned._

Your model hits 95% validation accuracy.

Nice. It works!

But does it?

We've all learned the routine: train on one set of data, validate on another, and see how well the model performs on examples it hasn't seen.

But here's the catch.

**Unseen data isn't necessarily unfamiliar data.**

Imagine training a CNN to recognize cats. It gets 95% accuracy on your validation images. But what if those images look remarkably similar to the training ones? Same breeds, similar backgrounds, familiar camera angles.

Now show it a hairless cat. Or one photographed in infrared.

Still confident about that 95%?

This got me thinking: **Are we measuring how well a model has learned, or how comfortable it is with the data we're testing it on?**

What started as a simple question about validation accuracy turned into a rabbit hole of K-Fold, stratification, data leakage, and a few surprising experiments.

The biggest surprise?

**I could make my model look smarter without making it any smarter.**

All I had to do was change how I validated it.

---

## 1. One validation split? That's a lot of trust.

My first thought was simple.

What if the 20% of data I held out happened to be unusually easy?

I'd get a great score, feel pretty good about my CNN, and never realize I'd been lucky.

Enter **K-Fold Cross Validation**.

Split the dataset into five parts. Train five fresh models, each time validating on a different part. Average the results.

Now every image gets its turn in validation.

Then I learned about **Stratified K-Fold**.

Suppose 90% of your images are cats and 10% are dogs. Random folds might have very different class proportions. Stratification keeps that 90:10 ratio approximately consistent.

Five rounds. Balanced folds. Less dependence on one lucky split.

Problem solved?

Not quite.

**Takeaway: K-Fold reduces our dependence on one split. Stratification balances class proportions. Neither guarantees the validation data is genuinely unfamiliar.**

> **Under the hood: the split isn't the only source of luck.**
> The random seed is too. We trained the exact same Xception41 recipe, on the exact same split, with four different seeds. Scores on our composite metric ranged from **49.9 to 51.5**.
> Our "best" model was simply the luckiest seed. The other three averaged **50.3**.
> So we stopped trusting single runs. A change was kept only if it beat the current best in at least 90% of 500 paired bootstrap resamples of the validation set.
> What that means: we build a "new" validation set of the same size by drawing images at random *with replacement*, so some images appear twice and some not at all. Both models are scored on that *same* resampled set (that's the "paired" part). Repeat 500 times. If the new model wins at least 450 of them, the improvement probably isn't down to which images happened to land in validation.
> Borderline wins got a second seed before we believed them.
> Several changes that looked like +0.6 to +1.0 on the first seed shrank to +0.1 to +0.4 on the second.

And that led to my next question.

_What if my model had already seen something almost identical to the validation image?_

## 2. Unseen image. Familiar information.

Imagine taking ten photos of the same cat, seconds apart.

Nine go into training. One goes into validation.

Technically, the tenth photo is unseen.

But the model has already seen that cat, in that room, under almost identical lighting.

Is that really a test of generalization?

This stopped being hypothetical during an image-classification project.

We discovered near-duplicate images in our dataset. Some were even different versions of the same original photograph.

Using perceptual hashing (a way to identify visually similar images), we grouped these look-alikes together.

Then we checked our standard Stratified 5-Fold split.

**8.4% of validation images had a near-duplicate in training.**

Wait. What?

We'd carefully separated the images, but not necessarily the information inside them.

So we tried **Stratified Group K-Fold**: keep related images together, never on opposite sides of a training-validation split.

For one CNN experiment:

- Conventional stratified split: **0.915 AUC**
    
- Group-aware split: **0.893 AUC**
    

AUC measures how well a classifier ranks positive examples above negative ones. Higher is better.

The comparison wasn't perfectly controlled because the validation sets changed too. But it was enough to challenge my confidence.

The model hadn't suddenly become worse.

We'd changed what we were asking it to prove.

**Takeaway: A validation image can be new without containing much new information. Preventing data leakage matters as much as choosing the right validation technique.**

> **Under the hood: how we found the look-alikes.**
> Every image got a 64-bit **difference hash (dHash)**: shrink it, compare neighbouring pixel brightness, and record each comparison as one bit.
> Two images whose hashes differed in **6 bits or fewer** (Hamming distance ≤ 6) were merged into the same group.
> Groups were then assigned to train, validation, or holdout as a whole. The split script asserts that no group appears on both sides, so a leak fails loudly instead of silently.
> The 8.4% figure came from a standard `StratifiedKFold(5, shuffle=True, random_state=42)`, the kind of split you'd write without thinking twice.
> One more caveat on 0.915 vs 0.893: the random-split model also trained on more images (5,600 vs 4,932), so the gap mixes "less leakage" with "less data". That's part of why I call the comparison imperfect.

But then another thought occurred to me.

_Even after removing look-alikes, aren't we still validating on the same kinds of images?_

## 3. What if the entire _kind_ of data is new?

Back to our cat classifier.

Suppose it has seen hundreds of Persian, Siamese, and tabby cats.

You validate it on different photographs of those same breeds.

Great accuracy.

Now introduce a breed it has never encountered.

That's a different challenge.

A random split mostly asks: _Can you recognize more examples drawn from a familiar distribution?_

I wanted to ask: **Can you handle something meaningfully different?**

In our project, we didn't have labels describing every underlying image-generation method. So we tried an approximation.

We used a pretrained CNN to turn images into feature vectors, clustered visually similar images, and held out entire clusters during training.

In other words, we deliberately kept certain _kinds_ of images unfamiliar.

For one model, the results were striking:

|Validation approach|AUC|
|---|---|
|Conventional stratified split|0.915|
|Near-duplicates grouped|0.893|
|Entire image clusters held out|**0.848**|

Same model architecture and training recipe. Very different evaluation conditions.

Those held-out clusters weren't guaranteed to represent truly new image-generation methods. But they gave us a harder, useful approximation of unfamiliar data.

And here's what surprised me most.

Some CNN architectures that looked strongest on familiar validation images performed substantially worse on unfamiliar clusters.

**The model with the highest familiar-data score wasn't necessarily the model I'd want to trust.**

Takeaway: **Don't just ask how well a model performs on unseen examples. Ask how different those examples are from what it learned.**

> **Under the hood: building "unfamiliar" on purpose.**
> We ran every image through an ImageNet-pretrained ResNet-50 and kept its feature vector. Then we ran **k-means with 16 clusters, separately for real and fake images**.
> Two whole clusters per class became the "unfamiliar" validation set, and one more per class went into the final holdout. Near-duplicate groups were kept in a single cluster, so look-alikes couldn't sneak across this boundary either.
> Here is the comparison behind the claim above. Each model got the same 6-minute training budget:
>
> | Model | Familiar AUC | Unfamiliar-cluster AUC |
> |---|---|---|
> | SE-ResNeXt-50 | **0.895** | 0.803 |
> | ConvNeXt-Tiny | 0.891 | 0.790 |
> | Xception41 | 0.893 | **0.848** |
> | EfficientNet-B0 | 0.876 | 0.825 |
>
> On familiar data the models sit within 2 points of each other. On unfamiliar clusters they spread over 6 points, and the ranking reshuffles.
> We also reimplemented a well-tuned Swin-Tiny recipe. It scored **0.925 AUC on a random split**. On held-out clusters it dropped to **0.799**.

Naturally, the next thought was:

_Could we teach the model to handle more variation?_

## 4. More augmentation must help. Right?

Data augmentation sounded like the obvious answer.

Flip images. Blur them. Add noise. Change compression. Randomly erase parts.

Give the model more variety so it can't memorize exact pixels.

Makes sense.

Except our experiments didn't always agree.

Random erasing? Worse.

Aggressive JPEG augmentation? Worse.

Some blur, noise, and downscaling? Small improvements.

Even more surprising: **training longer consistently made several of our models perform worse on the evaluation we cared about.**

I had assumed more training meant more learning.

Apparently, it can also mean more opportunities to fit the peculiarities of the training data.

One of our biggest improvements came from something less glamorous: adjusting the learning rate.

For an EfficientNet experiment, increasing the learning rate from 0.0003 to 0.008 improved held-out-cluster AUC from roughly 0.72 to 0.83.

That doesn't mean higher learning rates are always better. It means our original setting wasn't working well for this particular training setup.

**Takeaway: More data transformations, more epochs, and more complexity aren't automatically better. Every improvement needs to earn its place on meaningful validation data.**

> **Under the hood: what "worse" and "longer" actually meant.**
> Every experiment got a fixed **6-minute training budget**, so ideas competed on equal compute. An automated loop ran 55 of them, one change at a time.
>
> - **Random erasing:** worse at two different learning rates.
> - **JPEG re-compression (quality 30–95):** worse overall.
> - **One random blur, noise, downscale, or pixelation per image:** positive on two seeds, but never past our significance bar.
> - **Training 4–5× longer:** Xception41 went from 4 epochs to 16 and its unfamiliar-cluster AUC fell from 0.848 to 0.818. EfficientNet-B0 went from 5 epochs to 18 and fell from 0.832 to 0.806. The final models were retrained for the epoch counts that had been validated, not "as long as possible".
>
> Why did a higher learning rate help so much? At lr 0.001, training loss ended near **0.03**: the model had essentially memorised the training set. At 0.008 it ended near **0.34**, and the validation scores jumped. Larger steps kept the model from settling into the training data. At 0.012 it got worse again, so the effect has a ceiling.
>
> **My favourite failure: the model that learned nothing.**
> We tried *consistency training*: show the model a clean image and a degraded copy, and penalise it when the two predictions differ.
> One run scored dramatically worse than everything around it. Its training loss had settled at **1.32**.
> That number was the clue. A model that answers "50% fake" for *every* image has a loss of 2 × ln 2 ≈ **1.39** in our setup. Its two predictions always match, so the consistency penalty is zero.
> The model had found the laziest way to obey the rule: stop discriminating at all.
> Ramping the consistency penalty up gradually over the first two epochs fixed it, and the same configuration recovered to one of our best results.
> The scary part: when we added that broken model to an ensemble, the ensemble's validation score went *up* by 0.2. Small ensemble gains can be pure noise.

But there was still something bothering me.

_What if the model handles unfamiliar images, but fails when those images are blurry, compressed, or otherwise degraded?_

## 5. Robustness: Another way to be fooled

Imagine a cat classifier that works beautifully on clear photographs.

Now send it a blurry WhatsApp image.

Or a heavily compressed screenshot.

Still a cat. But can the model recognize it?

We tested this idea using image corruptions: blur, noise, compression, pixelation, brightness changes, and more.

Then came an important distinction.

If I train a model using JPEG-compressed images and validate it on JPEG compression, am I really testing its ability to handle unfamiliar damage?

Not entirely. I've already exposed it to that kind of transformation.

So we separated corruptions into two groups:

- **Seen corruptions:** Types allowed during training augmentation.
    
- **Unseen corruptions:** Types deliberately excluded from training.
    

Our final model achieved:

|Evaluation|AUC|
|---|---|
|Clean images|0.908|
|Seen corruptions (average)|0.905|
|Unseen corruptions (average)|**0.899**|

Encouraging!

But one transformation, center-crop zoom, dropped AUC to 0.862.

That told us something a single overall score couldn't: the model was relatively robust to several kinds of image degradation, but more sensitive to changes in framing.

**Takeaway: Generalization isn't one number. A model may handle new examples well but struggle with new conditions.**

> **Under the hood: the two lists, and a near-miss.**
>
> | Seen (allowed in training) | Unseen (never used in training) |
> |---|---|
> | JPEG compression, Gaussian blur, Gaussian noise, downscale-upscale, brightness/contrast, occlusion boxes, pixelation, grayscale | WebP compression, motion blur, gamma curves, posterize, center-crop zoom, median filter, a "social media" chain (downscale → JPEG → upscale) |
>
> Our composite score weighted unseen corruptions at **60%** and seen ones at 40%, so a model couldn't win just by mastering its own augmentations.
> The separation is easier to break than it sounds. At one point we tried stacking two random seen corruptions per training image. Downscale followed by JPEG is two "allowed" operations, and together they would have quietly recreated the held-out social-media chain. We caught it before the run and blocked that combination.
> On validation the weakest variant was usually **pixelation** (around 0.81–0.83 AUC). On the final holdout it was center-crop zoom. The weak spot moved, which is one more reason to look at per-corruption scores rather than an average.

And even after all this, one last question remained.

_What if the model is getting the right answers for the wrong reasons?_

## 6. What is the model actually looking at?

Remember the cat classifier?

Imagine discovering it mostly looks at green grass because so many training cats were photographed outdoors.

High accuracy. Completely wrong shortcut.

For our image classifier, we used **Grad-CAM**, a technique that highlights image regions influencing a CNN's prediction.

We wanted to see whether the model focused on meaningful facial regions or irrelevant artifacts like image borders.

Most of the highlighted evidence appeared around facial features, which was reassuring.

But there was a funny moment.

Our first heatmaps seemed to suggest a suspicious focus on image edges.

After investigating, we realized we'd been explaining the wrong output class for some images.

The explanation itself was misleading!

Fixing the explanation target changed the interpretation.

That was another lesson I hadn't expected.

**Takeaway: Don't blindly trust your model's predictions. But don't blindly trust the tools you use to explain those predictions either.**

> **Under the hood: what Grad-CAM is, and what "the wrong output class" means.**
> **Grad-CAM in plain words:** it paints a heatmap over the image. Warm colours mark the regions that pushed the model toward a particular answer, and cool colours mark regions that hardly mattered.
> **How it works, roughly:** a CNN's last convolutional layer holds a stack of small feature maps, coarse grids that each light up for some pattern (an eye corner, a skin texture, an edge) at each location in the image. Grad-CAM asks, for each map, *"if this map got a little stronger, how much would the score for answer X go up?"* That sensitivity is the gradient. It weights each map by its gradient, adds them together, keeps only the positive evidence, and stretches the small grid back to the full image size.
> The key detail is "answer X": **you have to choose which output to explain.** The heatmap shows evidence *for that answer*, not "what the model looked at" in general.
> Our models output a single "fake" score. Our first pass always explained that fake score, even for images the model had called *real*. On those images we were effectively asking "where is the evidence for the answer you rejected?" Averaged over 150 real faces, roughly a quarter of the heatmap sat in a band along the bottom edge, which looked exactly like a border shortcut. Explaining the class the model actually predicted made that edge pattern disappear.
> With the target fixed, **83–89% of the attention fell on the central face region**, and only 2–6% on the bottom edge.
> We also averaged Grad-CAM over 150 real and 150 fake holdout images. The faces are pre-aligned, so a class-average map is meaningful, and much harder to cherry-pick than a few nice examples.
> A bonus finding: the four models in our final ensemble looked at *different* places. One focused on the face boundary, one on the mouth and chin, one on the eyes, one on a tight central region. That's a plausible reason why averaging them helped.

---

## So, how well does your model _really_ generalize?

I started with what seemed like a straightforward question:

_Is my validation accuracy good?_

Now I find myself asking a few different ones.

Is the score consistent across different splits?

Are similar examples leaking between training and validation?

Does the model handle genuinely unfamiliar kinds of data?

Does it survive transformations it wasn't explicitly trained on?

And is it learning meaningful features or convenient shortcuts?

None of these tests guarantees real-world generalization. Even our hardest validation sets are only approximations of what a model might encounter later.

But together, they make it much harder to fool ourselves.

For our project, the final untouched holdout gave us **0.908 AUC**. A useful result, but not a promise that the model will perform equally well on every new dataset.

> **Under the hood: keeping the holdout honest.**
> The holdout was 589 images that no experiment, threshold choice, or model selection ever touched. We allowed ourselves at most three looks at it over the whole project and used two.
> Even so, 589 images is small. A standard-error estimate puts 0.908 at roughly **±0.025** (95% interval), so differences smaller than that between two models on this holdout shouldn't be over-read.

And that's probably my biggest learning from this whole rabbit hole.

**Validation is a measurement. Generalization is what we're trying to understand.**

The quality of the measurement depends on the questions we ask and the data we choose to ask them with.

These days, when I see a really impressive validation score, I'm still excited.

I'm just a little more curious about what went into it.

**Because sometimes the most useful experiment isn't the one that makes your score go up.**

**It's the one that tells you why it shouldn't have been that high in the first place.**

---

_These reflections came out of experiments during a deepfake-detection hackathon, but the lessons apply to almost any classification problem. Still learning, still experimenting, and definitely a lot more suspicious of beautiful validation scores than I used to be._

---

### Appendix: the setup, for the curious

- **Data:** 7,000 labelled 224×224 face images, real vs fake, roughly balanced. Split into 4,932 train, 1,479 validation (561 familiar + 918 from held-out clusters), and 589 untouched holdout.
- **Composite validation score:** 50 × (2·AUC_main − 1) + 20 × (2·AUC_robust − 1). AUC_main averages familiar and unfamiliar-cluster AUC. AUC_robust weights unseen corruptions 60% and seen ones 40%.
- **Search:** 55 automated experiments, each changing one thing under a 6-minute training budget. A change was kept only if it won at least 90% of 500 paired bootstrap resamples.
- **Final model:** an average of four ImageNet-pretrained CNNs (two Xception41 seeds, DenseNet-121, EfficientNet-B0) with horizontal-flip test-time averaging, retrained for their validated epoch counts (4, 4, 5 and 8).

#MachineLearning #DeepLearning #Generalization #CrossValidation #StudentLearning