from modules.preprocessing import rectify_sheet
from modules.detect_exam_code import read_exam_code_by_text
from modules.detect_answers import detect_bubbles_and_choices
from modules.grading import load_answer_bank_json, score_by_exam_code

def main():
    path_img = "data/samples/phieu_01.jpg"
    answer_bank = load_answer_bank_json("data/answers.json")

    img = cv2.imread(path_img)
    sheet = rectify_sheet(img)
    exam_code, _ = read_exam_code_by_text(sheet)
    choices = detect_bubbles_and_choices(sheet)
    correct, total, score = score_by_exam_code(exam_code, choices, answer_bank)
    print(f"[{exam_code}] {correct}/{total} → {score:.2f} điểm")

if __name__ == "__main__":
    main()
