# Reading list

> Papers, books, and videos referenced across the project. Numbered by sprint week — read item N before tackling Week N's math + build.

## Week 1 — Brownian motion, Itô calculus, GBM

1. Shreve, *Stochastic Calculus for Finance II: Continuous-Time Models*, ch. 3–5. **Primary text.**
2. Øksendal, *Stochastic Differential Equations*, ch. 2–5. Alternative if Shreve feels slow.
3. Lawler, [*Stochastic Calculus and Brownian Motion*](https://www.math.uchicago.edu/~lawler/finbook.pdf). Free, concise.
4. Video: [Brownian motion and Wiener processes explained](https://www.youtube.com/watch?v=ksrcU-foiRU).
5. Video: [Itô's Lemma From Scratch](https://www.youtube.com/watch?v=ZXsqxRRcH6g).

## Week 2 — Heston, Monte Carlo VaR

6. Heston, *A Closed-Form Solution for Options with Stochastic Volatility*, Review of Financial Studies, 1993.
7. Lord, Koekkoek, van Dijk, *A Comparison of Biased Simulation Schemes for Stochastic Volatility Models*, 2010 — full-truncation Euler analysis.
8. Andersen, *Efficient Simulation of the Heston Stochastic Volatility Model*, 2008.
9. Glasserman, *Monte Carlo Methods in Financial Engineering*, ch. 9 (VaR).
10. Artzner, Delbaen, Eber, Heath, *Coherent Measures of Risk*, Mathematical Finance, 1999.
11. McNeil, Frey, Embrechts, *Quantitative Risk Management*, ch. 2.

## Week 3 — Adjoint method, Neural SDEs, calibration losses

12. Li, Wong, Chen, Duvenaud, *Scalable Gradients for Stochastic Differential Equations*, AISTATS 2020. **Primary** for the adjoint.
13. Chen, Rubanova, Bettencourt, Duvenaud, *Neural Ordinary Differential Equations*, NeurIPS 2018. Deterministic precursor.
14. Kidger, Foster, Li, Lyons, *Neural SDEs as Infinite-Dimensional GANs*, ICML 2021. Useful framing, future-work tier.
15. Aït-Sahalia & Kimmel, *Maximum likelihood estimation of stochastic volatility models*, JFE 2007.
16. Gretton et al., *A Kernel Two-Sample Test*, JMLR 2012 (MMD).
17. Székely & Rizzo, *Energy statistics: A class of statistics based on distances*, JSPI 2013.
18. [google-research/torchsde](https://github.com/google-research/torchsde) — implementation reference.

## Week 5 — VaR backtesting

19. Kupiec, *Techniques for verifying the accuracy of risk measurement models*, 1995.
20. Christoffersen, *Evaluating Interval Forecasts*, IER 1998.

## Future-work / parking-lot reading (Phase 3+)

21. Raissi, Perdikaris, Karniadakis, *Physics-Informed Neural Networks*, JCP 2019. Repo: [maziarraissi/PINNs](https://github.com/maziarraissi/PINNs).
22. Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021.
23. Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023.
24. Video: [The Physics of A.I.](https://www.youtube.com/watch?v=dRkehLL19Wo).
25. Video: [Chaos: The Science of the Butterfly Effect](https://youtu.be/fDek6cYijxI).
