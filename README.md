# Low-Resource Counterspeech Generation for Indic Languages: The Case of Bengali and Hindi
[[Paper: Low-Resource Counterspeech Generation for Indic Languages: The Case of Bengali and Hindi]](https://aclanthology.org/2024.findings-eacl.111/)

*Mithun Das, Saurabh Kumar Pandey, Shivansh Sethi, Punyajoy Saha, Animesh Mukherjee* \
**Indian Institute of Technology Kharagpur** \
[European Chapter of the Association for Computational Linguistics (EACL 2024)](https://2024.eacl.org/)

## Abstract

With the rise of online abuse, the NLP community has begun investigating the use of neural architectures to generate counterspeech that can “counter” the vicious tone of such abusive speech and dilute/ameliorate their rippling effect over the social network. However, most of the efforts so far have been primarily focused on English. To bridge the gap for low-resource languages such as Bengali and Hindi, we create a benchmark dataset of 5,062 abusive speech/counterspeech pairs, of which 2,460 pairs are in Bengali, and 2,602 pairs are in Hindi. We implement several baseline models considering various interlingual transfer mechanisms with different configurations to generate suitable counterspeech to set up an effective benchmark. We observe that the monolingual setup yields the best performance. Further, using synthetic transfer, language models can generate counterspeech to some extent; specifically, we notice that transferability is better when languages belong to the same language family.

**[Note]** Code release is in progress. Stay tuned!!

# Citation

## If you find our work useful, please cite using:
```
@inproceedings{das-etal-2024-low,
    title = "Low-Resource Counterspeech Generation for {I}ndic Languages: The Case of {B}engali and {H}indi",
    author = "Das, Mithun  and
      Pandey, Saurabh  and
      Sethi, Shivansh  and
      Saha, Punyajoy  and
      Mukherjee, Animesh",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.111",
    pages = "1601--1614",
    abstract = "With the rise of online abuse, the NLP community has begun investigating the use of neural architectures to generate counterspeech that can {``}counter{''} the vicious tone of such abusive speech and dilute/ameliorate their rippling effect over the social network. However, most of the efforts so far have been primarily focused on English. To bridge the gap for low-resource languages such as Bengali and Hindi, we create a benchmark dataset of 5,062 abusive speech/counterspeech pairs, of which 2,460 pairs are in Bengali, and 2,602 pairs are in Hindi. We implement several baseline models considering various interlingual transfer mechanisms with different configurations to generate suitable counterspeech to set up an effective benchmark. We observe that the monolingual setup yields the best performance. Further, using synthetic transfer, language models can generate counterspeech to some extent; specifically, we notice that transferability is better when languages belong to the same language family.",
}
```
