import javax.swing.*;
import java.awt.*;
import java.awt.Graphics;
import java.io.IOException;
import java.util.Scanner;

public class CirclesGrid  extends JPanel{
    private static final int width = 480;
    private static final int height = 480;
    private static final float[] gravity = {0.0f,-0.02f};
    private static final float timeStep = 1f/60f;
    private static final float minimum_radius = 0.05f;
    private static final float minimum_x = 0.1f;
    private static final float minimum_y = 0.1f;
    private static final int strategy = 1;
    private static final float restitution = 0.5f;
    private static final long sleepTime = 17L;
    private static final float minimum_mass = 4f;
    private static final float circleCount = 15;
    private float[][] circles;
    private Circles2D engine = new Circles2D(true);

    public void simulate(){
        circles = engine.loop();
        repaint();
    }

    private void init(){
        circles = new float[4][15];

        for(int i=5; i>0; i--){
            for(int j=i; j>0; j--){
                engine.addCircle(0.35f+0.1f*(float)j-0.05f*(float)i,0.6f-0.1f*(float)i,minimum_radius,minimum_mass);
            }
        }

        try {
            engine.init(timeStep, strategy, restitution, gravity);
        }
        catch(IOException e){
            System.err.println("Failed to find opencl file");
            e.printStackTrace();
        }
    }

    public CirclesGrid(){
        setPreferredSize( new Dimension( width + 20, height + 40) );
        setBackground( Color.BLACK );
        this.init();
        this.simulate();
    }

    protected void paintComponent( Graphics g ){
        super.paintComponent( g );
        Graphics2D g2d = (Graphics2D)g;
        for(int i=0; i<circleCount; i++){
            Color color = new Color( 0, (int)((205f/19f)*(circles[3][i]-1f)+(50f)), (int)((205f/19f)*(circles[3][i]-1f)+(50f)));
            g2d.setColor(color);
            g2d.fillOval((int) ((float) width * (circles[0][i] - circles[2][i])),
                    (int) ((float) height - (float) height * (circles[1][i] + circles[2][i])),
                    (int) (((float) width) * circles[2][i] * 2f), (int) (((float) height) * circles[2][i] * 2f));
        }

    }

    public static void runApplication(final JPanel app){
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                JFrame frame = new JFrame();

                frame.setSize( app.getPreferredSize() );
                frame.setTitle( app.getClass().getName() );
                frame.setDefaultCloseOperation( JFrame.EXIT_ON_CLOSE );
                frame.add( app );
                frame.setVisible( true );
            }
        });
    }

    public static void main(String[] args){
        CirclesGrid test = new CirclesGrid();
        CirclesTest.runApplication(test);
        Scanner sc = new Scanner( System.in );
        while(true){
            long start = System.currentTimeMillis();
            test.simulate();
            long end = System.currentTimeMillis();
            long runtime = end-start;
            try{
                Thread.sleep(sleepTime - runtime);
            }
            catch(Exception e){
                Thread.currentThread().interrupt();
            }
        }
    }
}
