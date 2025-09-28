# Introduction

Imagine planning a long road trip with friends. You don’t just look at the map—you also check the weather, traffic, and fuel stops. In the same way, when we plan to run heavy AI models, we don’t just look at the code—we look at the energy.

Renewable energy (like wind and solar) changes a lot from hour to hour, especially in countries like Germany. Some hours are rich with clean power. Other hours lean on fossil fuels. If we can see these patterns 72 hours ahead, we can choose better times to train big models—saving money, reducing emissions, and being kinder to the planet.

That’s the heart of this project: a 72-hour forecast for renewable energy percentage and carbon intensity. Our aim is not just accuracy, but usefulness—simple, reliable insights that tell us “when” is the best time to compute.

What makes this exciting?

- It connects AI with real-world energy trends.
- It helps schedule training when the grid is greenest.
- It opens the door to greener, cheaper, and smarter ML operations.

How do we do it? We use proven time-series models—like LSTMs and their seasonal cousin, CycleLSTM—to learn patterns in hourly data over the last five years. We also test modern transformer families for long-range patterns. The models predict the next 72 hours, and we evaluate them carefully with metrics that fit long horizons (beyond just MAE), so we don’t miss the bigger picture.

Why does this matter now? Because the need is immediate. AI workloads are growing, energy prices are volatile, and climate goals are urgent. With a clear, beginner-friendly forecast, teams can make smarter choices: pause during “dirty” hours, sprint during “clean” hours, and keep the same model quality with a lower footprint.

In short, think of this project as a weather report for clean energy. It tells us when the wind is on our side—so we can plan, train, and innovate more responsibly.
