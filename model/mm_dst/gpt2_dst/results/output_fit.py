import json

def convert_bart(line):
    line = line.replace("<pad><cls><cls>","").strip()
    line = line.replace(" <sep1> "," ")
    line = line.replace("<sep2>","<EOB>")
    line = " pad pad pad => Belief State : " + line
    return line

if __name__ == "__main__":
    input_f = "bart_test_predicted.txt"
    output_f = "bart_test_out.txt"
    lines = open(input_f,"r",encoding="utf8").readlines()
    output_lines = []
    for line in lines:
        output_lines.append(convert_bart(line))
    # print(output_lines)
    with open(output_f,"w",encoding="utf8") as f:
        f.write("\n".join(output_lines))