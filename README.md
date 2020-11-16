# apml_ex1

Link to data sheets: https://docs.google.com/spreadsheets/d/1_04Bag4r5kpfOpma2Mi7TeXJrahfNGbUj6TKe7vBAGM/edit?usp=sharing



Adversarial Examples

The code for it can be found in the module ‘main.py’.
The function findAdversarialExample contains the logic of assembling the needed pieces, running the attack loop, and plotting the results.
The function fgsm_attack serves as the attacker.
The function _findAdversarialExampleInternal contains the main attack loop - it iterates over the data set, and tries to use the attacker in order to generate an adversarial example.

