# Generated Stories

Sample outputs from the trained model. Each story was generated from a structured prompt and the model's `generate` function with `temperature=0.8`, `top_k=40`.

---

## Sample 1 — Cat and bird

**Prompt:**
- Features: Dialogue
- Words: cat, fence, bird
- Summary: A cat watches a bird from a fence.

**Output:**
> A cat lived in a house. The cat liked to play with a ball. One day, the cat saw a bird with a hurt wing. The bird wanted help.
>
> "Can I help you, cat?" asked the cat.
> "Sure, I can," said the bird.
>
> The cat was happy. They played together all day long.

---

## Sample 2 — Sharing toys

**Prompt:**
- Features: MoralValue
- Words: share, friend, toy
- Summary: A girl learns to share her toys with a friend.

**Output:**
> Once upon a time, there was a little girl who had a toy. The girl had a toy. It was a very nice toy, and the toy was a little girl who loved it very much. 
One day, the girl wanted to share her toys with her friend. She went to her friend's house to play with her. When she got home, the girl was so happy, she hugged her toy tightly. 
The little girl had brought all her toys back and forth to play with them. She was so glad that she had her favorite toy. 
The little girl played with her toys every day and had lots of fun. She even found a new toy inside her own box. She was so happy and grateful to have a new toy. The end.

---

## Sample 3 — Lost balloon

**Prompt:**
- Features: Twist
- Words: park, balloon, lost
- Summary: A child loses a balloon at the park.

**Output:**
> Once upon a time, there was a little town. In the town there was a new friend. The new boy had something in his hands. He looked everywhere he went.
He asked his mum for help. He saw all the things he could find, but he was careless. Then he had an idea. He went to the park and found his balloon. It had a little boy with a kite.
The boy tried to get the new balloon. But he couldn't. He started to feel sad. He looked for someone to help.
Then, he noticed a small boy. He asked in a big smile! The boy said, "What is your turn?" The boy said that he was just a boy. He was so

---

## Observations

- The model follows the prompt's words and topic well in the first 1–2 paragraphs.
- Dialogue tagging (e.g., `"... said the cat`) is consistent with the training data.
- Stories drift after roughly 150 generated tokens — characters may shift or new characters appear that weren't in the prompt. This is the standard small-model limitation.
- Lowering temperature (0.5–0.7) produces more focused output at the cost of variety.