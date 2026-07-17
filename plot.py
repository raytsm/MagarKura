import re
import matplotlib.pyplot as plt
from collections import defaultdict

metrics = defaultdict(list)
epochs = []

with open("train.log", "r") as f:
    for line in f:

        # Training summary line
        if "| train | epoch" in line:
            epoch = int(re.search(r"epoch\s+(\d+)", line).group(1))
            epochs.append(epoch)

            def grab(key):
                m = re.search(fr"{key}\s+([\d.e+-]+)", line)
                return float(m.group(1)) if m else None

            metrics["train_loss"].append(grab("loss"))
            metrics["train_ppl"].append(grab("ppl"))
            metrics["lr"].append(grab("lr"))
            metrics["gnorm"].append(grab("gnorm"))
            metrics["loss_scale"].append(grab("loss_scale"))
            metrics["wps"].append(grab("wps"))

        # Validation summary line
        if "| valid | epoch" in line:
            metrics["valid_loss"].append(
                float(re.search(r"loss\s+([\d.]+)", line).group(1))
            )
            metrics["valid_ppl"].append(
                float(re.search(r"ppl\s+([\d.]+)", line).group(1))
            )

            bleu = re.search(r"bleu\s+([\d.]+)", line)
            metrics["bleu"].append(float(bleu.group(1)) if bleu else None)


def save_plot(x, y, title, ylabel, fname, labels=None):
    plt.figure()
    if isinstance(y, list) and labels:
        for yi, lab in zip(y, labels):
            plt.plot(x, yi, label=lab)
        plt.legend()
    else:
        plt.plot(x, y)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


# 1–3: Loss plots
save_plot(epochs, metrics["train_loss"],
          "Training Loss vs Epoch", "Loss", "01_train_loss.png")

save_plot(epochs, metrics["valid_loss"],
          "Validation Loss vs Epoch", "Loss", "02_valid_loss.png")

save_plot(epochs,
          [metrics["train_loss"], metrics["valid_loss"]],
          "Training vs Validation Loss", "Loss",
          "03_train_vs_valid_loss.png",
          labels=["Train", "Validation"])


# 4: BLEU
save_plot(epochs, metrics["bleu"],
          "BLEU Score vs Epoch", "BLEU", "04_bleu_vs_epoch.png")


# 5: Perplexity
save_plot(epochs,
          [metrics["train_ppl"], metrics["valid_ppl"]],
          "Perplexity vs Epoch", "PPL",
          "05_ppl_vs_epoch.png",
          labels=["Train", "Validation"])


# 6: Learning rate
save_plot(epochs, metrics["lr"],
          "Learning Rate Schedule", "Learning Rate",
          "06_lr_schedule.png")


# 7: Gradient norm
save_plot(epochs, metrics["gnorm"],
          "Gradient Norm vs Epoch", "Gradient Norm",
          "07_gradient_norm.png")


# 8: Loss scale (FP16)
save_plot(epochs, metrics["loss_scale"],
          "Loss Scale vs Epoch (FP16)", "Loss Scale",
          "08_loss_scale.png")


# 9: Throughput
save_plot(epochs, metrics["wps"],
          "Training Throughput (WPS)", "Words per Second",
          "09_wps.png")


print("✅ All Fairseq plots generated successfully.")
