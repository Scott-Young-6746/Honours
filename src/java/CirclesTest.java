import javax.swing.*;
import java.awt.*;
import java.awt.Graphics;
import java.io.IOException;
import java.util.Scanner;

public class CirclesTest  extends JPanel{
    private static final int width = 480;
    private static final int height = 480;
    private static final float[] gravity = {0.0f,-0.01f};
    private static final float timeStep = 1f/60f;
    private static final int numberOfCircles = 7500;
    private static final float minimum_radius = 0.01f;
    private static final float radius_range = 0.005f;
    private static final float minimum_x = 0.1f;
    private static final float x_range = 0.8f;
    private static final float minimum_y = 0.1f;
    private static final float y_range = 0.8f;
    private static final int strategy = 1;
    private static final float restitution = 1f;
    private static final long sleepTime = 17L;
    private static final float minimum_mass = 1f;
    private static final float mass_range = 9f;
    
    private float[][] circles;
    private Circles2D engine = new Circles2D(true);
    private  int circleCount;

    public void simulate(){
        long start = System.currentTimeMillis();
        circles = engine.loop();
        long end = System.currentTimeMillis();
        long runtime = end-start;
        System.out.println("took " + runtime + " milliseconds to loop");
        repaint();
    }

    private void init(int circleCount){
        this.circleCount = circleCount;
        circles = new float[4][circleCount];

        for(int i=0; i<circleCount; i++){
            engine.addCircle((float)(x_range*Math.random())+minimum_x,
                             (float)(y_range*Math.random())+minimum_y,
                             (float)(radius_range*Math.random())+minimum_radius,
                             (float)(mass_range*Math.random())+minimum_mass);
        }

        try {
            engine.init(timeStep, strategy, restitution, gravity);
        }
        catch(IOException e){
            System.err.println("Failed to find opencl file");
            e.printStackTrace();
        }
    }

    public CirclesTest(int circleCount){
        setPreferredSize( new Dimension( width + 20, height + 40) );
        setBackground( Color.BLACK );
        this.init(circleCount);
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
        CirclesTest test = new CirclesTest(numberOfCircles);
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
