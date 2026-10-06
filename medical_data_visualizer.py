import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 1. Import the data
df = pd.read_csv("medical_examination.csv")


# 2. Add an overweight column
df["overweight"] = (
    df["weight"] / ((df["height"] / 100) ** 2) > 25
).astype(int)


# 3. Normalize cholesterol and gluc
df["cholesterol"] = df["cholesterol"].apply(
    lambda x: 0 if x == 1 else 1
)

df["gluc"] = df["gluc"].apply(
    lambda x: 0 if x == 1 else 1
)


# 4. Draw Categorical Plot
def draw_cat_plot():
    
    # 5. Create DataFrame for cat plot
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=[
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "overweight"
        ]
    )

    # 6. Group and reformat the data
    df_cat = (
        df_cat
        .groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # 7. Create the categorical plot
    fig = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar"
    )

    # 8. Get the figure
    fig = fig.fig

    # 9. Do not modify
    fig.savefig("catplot.png")

    return fig


# 10. Draw Heat Map
def draw_heat_map():

    # 11. Clean the data
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"]) &
        (df["height"] >= df["height"].quantile(0.025)) &
        (df["height"] <= df["height"].quantile(0.975)) &
        (df["weight"] >= df["weight"].quantile(0.025)) &
        (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # 12. Calculate correlation matrix
    corr = df_heat.corr()

    # 13. Generate mask for upper triangle
    mask = pd.DataFrame(
        [[True if j >= i else False
          for j in range(corr.shape[1])]
         for i in range(corr.shape[0])],
        index=corr.index,
        columns=corr.columns
    )

    # 14. Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Plot correlation matrix
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    # 16. Do not modify
    fig.savefig("heatmap.png")

    return fig