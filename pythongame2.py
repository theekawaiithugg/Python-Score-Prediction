import csv
import datetime
import os
import pygame, sys
import random
import pyttsx3
import pandas as pd
import time

# Set the current working directory to the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# Check and create the initial CSV if not exists
if not os.path.exists("snake_scores.csv"):
    with open("snake_scores.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "PlayerName", "Score", "SnakeLength", "DistanceToFood", "ScoreIncrease","GameDuration"])  # Example header

expected_fields = 7  # Adjust based on the number of expected fields
clean_rows = []

# Now read the existing data for cleaning
if os.path.exists("snake_scores.csv"):
    with open("snake_scores.csv", 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == expected_fields:
                clean_rows.append(row)
    # Write to the cleaned file (this creates the file if it doesn't exist)
    with open("snake_scores_clean.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(clean_rows)

# Load data with pandas, handle bad lines
data = pd.read_csv("snake_scores.csv", on_bad_lines='skip')

def say_name(name):
    engine = pyttsx3.init()
    engine.say(f"Hello {name}, welcome to snake, your score will be recorded for Artificial Intelegence!")
    engine.runAndWait()


def get_player_name(game_window, font):
    name = ""
    prompt_text = "Hello! May I have your name? "  # The prompt text
    input_box = pygame.Rect(50, 100, 140, 40)  # Position and size of the input box
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color = color_inactive
    active = False  # Input box is not active until clicked

    while True:
        game_window.fill((30, 30, 30))  # Fill the window with a background color
        # Render and display the prompt text
        prompt_surface = font.render(prompt_text, True, pygame.Color('white'))
        game_window.blit(prompt_surface, (40, 20))  # Adjust the position as needed
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                color = color_active if active else color_inactive
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        return name
                    elif event.key == pygame.K_BACKSPACE:
                        name = name[:-1]
                    else:
                        name += event.unicode

        
        txt_surface = font.render(name, True, color)
        game_window.blit(txt_surface, (input_box.x+5, input_box.y+5))
        pygame.draw.rect(game_window, color, input_box, 2)

        pygame.display.flip()




class Snake:
    def __init__(self):
        self.body = [[100, 50], [90, 50], [80, 50]]
        self.direction = 'RIGHT'
        self.change_to = self.direction

    def change_direction(self):
        if self.change_to == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.change_to == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.change_to == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.change_to == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'

    def move(self):
        if self.direction == 'UP':
            self.body.insert(0, [self.body[0][0], self.body[0][1] - 10])
        if self.direction == 'DOWN':
            self.body.insert(0, [self.body[0][0], self.body[0][1] + 10])
        if self.direction == 'LEFT':
            self.body.insert(0, [self.body[0][0] - 10, self.body[0][1]])
        if self.direction == 'RIGHT':
            self.body.insert(0, [self.body[0][0] + 10, self.body[0][1]])
        self.body.pop()

class Food:
    def __init__(self, game_window_width, game_window_height):
        self.position = [random.randrange(1, (game_window_width//10)) * 10, random.randrange(1, (game_window_height//10)) * 10]
        self.is_eaten = False

    def spawn_new_food(self, game_window_width, game_window_height):
        if self.is_eaten:
            self.position = [random.randrange(1, (game_window_width//10)) * 10, random.randrange(1, (game_window_height//10)) * 10]
            self.is_eaten = False

class Game:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.game_window = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.snake = Snake()
        self.start_time = None  # Initialize start time
        self.food = Food(self.width, self.height)
        self.score = 0
        self.my_font = pygame.font.SysFont('times new roman', 80)
        self.reset()

    def reset(self):
            self.snake = Snake()
            self.food = Food(self.width, self.height)
            self.score = 0
            self.start_time = time.time()  # Start timer when game resets
  
    def game_over(self):
        game_duration = time.time() - self.start_time  # Calculate game duration
        game_duration = round(game_duration, 2)  # Round duration to 2 decimal places
        
        go_surface = my_font.render('Your Score is : ' + str(self.score), True, 'red')
        go_rect = go_surface.get_rect()
        go_rect.midtop = (self.width/2, self.height/4)
        self.game_window.fill('black')
        self.game_window.blit(go_surface, go_rect)
        pygame.display.flip()
        # Calculate snake_length as the length of the snake's body
        snake_length = len(self.snake.body)
        
        # Calculation for distance_to_food
        # This is a Euclidean distance. 
        snake_head = self.snake.body[0]
        food_position = self.food.position
        distance_to_food = ((snake_head[0] - food_position[0])**2 + (snake_head[1] - food_position[1])**2) ** 0.5
        
        # Assuming score_increase is the score (for simplification)
        score_increase = self.score  # Or however you calculate score increment
        
        #ADD THIS
        # Use 'a' mode for appending data to the CSV
        with open("snake_scores.csv", mode="a", newline='') as file:
            writer = csv.writer(file)
            # Check if the file is empty to write the header
            if file.tell() == 0:
                writer.writerow(["Timestamp", "PlayerName", "Score", "SnakeLength", "DistanceToFood", "ScoreIncrease", "GameDuration"])
            writer.writerow([datetime.datetime.now(), player_name, self.score, snake_length, distance_to_food, score_increase, game_duration])
            
        # Saved your data in a CSV with columns: snake_length, distance_to_food, score_increase
        # Existing game over logic to display score and log data
        # ...

        pygame.time.wait(2000)  # Wait for 2 seconds

        # Display a message asking the player if they want to play again or exit
        # For simplicity, assume pressing 'R' restarts and 'Q' quits
        restart_surface = my_font.render('Press R to Restart or Q to Quit', True, 'white')
        restart_rect = restart_surface.get_rect()
        restart_rect.midtop = (self.width/2, self.height/2)
        self.game_window.blit(restart_surface, restart_rect)
        pygame.display.flip()

        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # Quit the game
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_r:  # Restart the game logic here
                        self.reset()
                        waiting_for_input = False  # Could set a flag to restart the game loop

        
        

    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    self.snake.change_to = pygame.key.name(event.key).upper()

        self.snake.change_direction()
        self.snake.move()

        # Check food collision
        if self.snake.body[0] == self.food.position:
            self.score += 1
            self.snake.body.append(self.snake.body[-1])  # Grow the snake
            self.food.is_eaten = True

        self.food.spawn_new_food(self.width, self.height)

        # Check game over conditions
        if (self.snake.body[0][0] < 0 or self.snake.body[0][0] > self.width-10 or 
            self.snake.body[0][1] < 0 or self.snake.body[0][1] > self.height-10 or 
            self.snake.body[0] in self.snake.body[1:]):
            self.game_over()

        self.game_window.fill('black')
        for pos in self.snake.body:
            pygame.draw.rect(self.game_window, 'green', pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(self.game_window, 'white', pygame.Rect(self.food.position[0], self.food.position[1], 10, 10))

        pygame.display.update()
        self.clock.tick(10)


if __name__ == "__main__":
    pygame.init()
    my_font = pygame.font.SysFont('times new roman', 30)  # Use an appropriate size
    game_window = pygame.display.set_mode((600 , 300))
    player_name = get_player_name(game_window, my_font)
    say_name(player_name)  # Speak the player's name
    game = Game(800, 400)

    while True:
        game.play_step()
