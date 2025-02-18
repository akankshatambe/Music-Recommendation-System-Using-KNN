# Music-Recommendation-System-Using-KNN
Simple Analysis and Comparision of Deep Learning and Genetic Algorithm Techniques for Music Generation
Abstract
 Music Generation (MG) is an intriguing area of study due to the bridge it
 provides between the creative arts and AI .The goal is to teach a computer to pro
duce music that is both unlimited in variety and endlessly enjoyable to listen to.
 Melody, harmony, and rhythm are all essential musical components. The complex
 nature of musical creation presents significant difficulties for AI. In this study, we
 explore whether evolutionary algorithms and LSTM are well-suited to the job of
 music creation and if these systems can handle the complexities of music. Long
 short-term memory (LSTM) networks are able to successfully describe sequential
 data, whereas genetic algorithms are employed as a tool to integrate musical ideas
 that mimic the combination-theory of creativity. LSTM networks have been widely
 utilized to generate character-by-character sheet music. However, these LSTM
 models need considerable training time before they can generate syntactically valid
 and aesthetically pleasing sheet music. This work presents an analysis of solu
tions in general, as well as an examination of the solutions acquired in light of the
 most important factors. The music score includes suggested solutions. The results
 also reveal a difference in efficiency between various rating strategies and genetic
 operators.
 1 Introduction
 The use of algorithms to compose music using AI has been around since the early 1990s,
 and many different techniques have been put to use since then. Nonetheless, the idea
 that music is created according to predetermined laws and structural standards has been
 around for quite some time, perhaps inspiring the concept of teaching a system to create
 music without human input. For this challenge, we aren’t only interested in finding
 a way to create music digitally from a particular input sequence; rather, we want to
 look at the more difficult problem of creating new sequences that are both aesthetically
 pleasing, rhythmically interesting, and melodically engaging. Humans have an innate
 talent for creating beautiful music, and the ’intelligence’ component consists of tapping
 into that talent in a structured, rather than haphazard, manner. Two algorithms, a
 genetic algorithm and a Recurrent Neural network, will be used to carry out the task of
 algorithmic music composition. The effectiveness and complexity of all three methods
 will be compared. Since the human mind has a finite amount of originality when it comes
 to writing music, an algorithmic method is necessary if one is to think that intriguing
 1
musical patterns will emerge when the music is generated electronically. Perhaps there is
 a limit to the kind of art and music that can exist in human heads, whereas a computer is
 open to all possibilities. The goal of taking such an approach is to either find an entirely
 new musical genre or to create much superior works within the same genre. This project
 aims to apply Recurrent neural network algorithms to generate music given a repository
 of music to train on. The development of this project had the following structure:
 1. Preparing the input data– deciding the features to take into account
 2. Building the model and training on the input data
 3. Generating music with the model and evaluation of results
 The above steps were repeated for 2 different models and experiments which allows us to
 compare the results between different generative models:
 1. Single note prediction with a long short term memory network
 2. Multiple notes prediction with a long short term memory network
 3. Music generation using genetic algorithm
 Essentially, this process shares similarities with the human learning process, where a
 composer who has listened to music would probably be compelled to write music close
 to what has been heard. Additionally, we notice that the models have not received any
 formal information on music theory and notions of compositions, but instead learning it
 all from “listening” to real music; comparable to someone who learned composition by
 ear.
 (a) Representation of Chords
 1.1 Motivation
 In many aspects, music generation is quite similar to text generation; if we consider notes
 to be words, we can see that the process is very similar. This indicates that studying
 one can be beneficial to studying the other, and vice versa. The process of making music
 has always captivated me. We typically think of music as an exalted abstract form of
 artistic expression, but it is just a formal system with its own set of laws and regulations.
 Music that sounds ”good” is not simply an accident; there are reasons for it, theory
 behind it, and seeing an algorithm ”understand” the inner workings of such a language,
 deduce the laws of the system by observing it is quite intriguing and can 2 give insights
 that are beneficial in other domains. Many uses may be envisaged from a more practical
 standpoint. For example, we might use it in conjunction with computer vision to create
 background music that corresponds to the atmosphere of a film. We may conceive a use
 2
case for a type of human-machine collaborative composing. Even if music appears to be
 a relatively simple field to apply generative models to, there is a lot to be learned from it,
 from the nature of music itself to the learning process of something as seemingly abstract
 as music. I would also add that it provides an amazing tool and program for all the
 world’s dreamers and artists to see.
 Achivement
 This study successfully created music using neural networks through two methods: a basic
 recurrent neural network (RNN) Long-Term-Short-Term Model (LSTM) and a genetic
 Algorithm (GA). In the first half, we work using RNNs to create an LSTM model that can
 handle the task of music generating appropriately. LSTMs employ a sequence of gates to
 determine whether information is important to a given job. LSTM networks employ three
 main gates: forget, input, and output gates. When each of these gates is modified, one
 time step is completed. We will concentrate on the LSTM technique in this study since
 we can utilize these LSTM cells to create our model architecture from scratch after some
 pre-processing processes. Second, we may make music using the Genetic Algorithm (GA),
 which is usually used for model optimization. GA generates music by leveraging existing
 music. [2]. According to [2,] a genetic algorithm can emphasize the strong rhythm in
 each fragment and merge them into separate pieces of music. However, it is inefficient
 since each iteration step includes a latency. Furthermore, because of the lack of context,
 it is difficult to obtain coherence and deep-seated rhythm information.
 1.2 Relevance
 So, why is the project even relevant? Will our efforts be useful to others? Consider AIVA,
 which has built their entire business around this very technology [1]. Their target market
 includes game creators who wish to employ unique music for hours of gaming, but it is
 frequently too expensive for small studios to engage humans to create and produce this
 music. Instead, algorithms might be employed to provide more cost-effective solutions for
 all types of producers who desire original music for their projects. By further developing
 creative algorithms, it might be used to literature, film, and art, as well as other areas
 where a great number of personalized material is required. Algorithms incorporating
 creative aspects may also be created as a tool to enable more individuals to participate in
 creative processes without professional expertise. AI may become the new indispensable
 tool for artists in the future
 Research Question
 Time series data are part of the study work. We’re looking at melody as a time series.
 We frequently hear melodies and songs being repeated. Is it feasible to develop new
 compositions from old compositions using deep learning methods and other optimization
 algorithms?
 Q1) Is the Deep Learning model superior to the Genetic Algorithm for music generation?
 Which algorithm is most suited for music generation?
 Q2) What variables should be considered when evaluating the efficiency of music?
 3
1.3 Proposed Research Objevtives
 The following are the study’s objectives: Working on two Experiments is how the research
 is carried out and how the melody generated is optimized.
 • The first experiment’s focus is on working on the LSTM model to train the data to
 create the notes.
 In this example, we trained the model in three separate scenarios with varied epoch val
ues.
 • In the second experiment, we utilize a genetic algorithm to compose music.
 • In second following experiment The program takes a melody from an input midi file
 (*.mid) and uses genetic programming (evolutionary algorithm) to produce an accom
paniment consisting of numerous three note chords (triads) that would sound good when
 played with the input tune
